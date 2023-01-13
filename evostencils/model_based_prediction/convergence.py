import lfa_lab
import evostencils.stencils.multiple as periodic
import evostencils.stencils.constant as constant
from evostencils.ir import base, system, partitioning
from multiprocessing import Process, Queue
from typing import List


@periodic.convert_constant_stencils
def stencil_to_lfa(stencil: periodic.Stencil, grid: base.Grid):
    def recursive_descent(array, dimension):
        if dimension == 1:
            # TODO fix this
            return [lfa_lab.SparseStencil([(offset, complex(value)) for offset, value in element.entries])
                    for element in array]
        else:
            return [recursive_descent(element, dimension - 1) for element in array]

    tmp = recursive_descent(stencil.constant_stencils, stencil.dimension)

    ndarray = lfa_lab.NdArray(tmp)
    return lfa_lab.from_periodic_stencil(ndarray, grid)


def lfa_sparse_stencil_to_constant_stencil(stencil: lfa_lab.SparseStencil):
    return constant.Stencil(tuple(entry for entry in stencil), stencil.dim)


class ConvergenceEvaluator:

    def __init__(self, dimension, coarsening_factors, finest_grid):
        self._coarsening_factors = coarsening_factors
        self._dimension = dimension
        self._lfa_grids = [lfa_lab.Grid(dimension, g.spacing) for g in finest_grid]

    @property
    def lfa_grids(self):
        return self._lfa_grids

    def set_lfa_grids(self, new_lfa_grids: List[lfa_lab.Grid]):
        self._lfa_grids = new_lfa_grids

    def reinitialize_lfa_grids(self, finest_grid: List[base.Grid]):
        self._lfa_grids = [lfa_lab.Grid(self.dimension, g.spacing) for g in finest_grid]

    @property
    def coarsening_factors(self):
        return self._coarsening_factors

    @property
    def dimension(self):
        return self._dimension

    def get_lfa_grid(self, grid: base.Grid, i: int):
        lfa_grid = self.lfa_grids[i]
        step_size = lfa_grid.step_size()
        while step_size < grid.spacing:
            lfa_grid = lfa_grid.coarse(self.coarsening_factors[i])
            step_size = lfa_grid.step_size()
        return lfa_grid

    def transform(self, expression: base.Expression):
        if expression.lfa_symbol is not None:
            return expression.lfa_symbol
        if isinstance(expression, base.Cycle):
            correction = self.transform(expression.correction)
            if isinstance(expression.approximation, system.ZeroApproximation):
                approximation = correction.matching_zero()
            elif isinstance(expression.approximation, system.Approximation):
                approximation = correction.matching_identity()
            else:
                approximation = self.transform(expression.approximation)
            tmp = approximation + expression.relaxation_factor * correction
            if expression.partitioning == partitioning.Single:
                result = tmp
            elif expression.partitioning == partitioning.RedBlack:
                if isinstance(expression.correction, base.Multiplication):
                    operand1 = expression.correction.operand1
                    operand2 = expression.correction.operand2
                    if isinstance(operand1, base.Inverse) and isinstance(operand2,
                                                                         base.Residual):
                        red_entries = []
                        black_entries = []
                        operator = operand1.operand
                        while not isinstance(operator, system.Operator):
                            if isinstance(operator, base.UnaryExpression):
                                operator = operator.operand
                            else:
                                raise RuntimeError("Computation could not be partitioned.")
                        for i, row in enumerate(operator.entries):
                            red_entries.append([])
                            black_entries.append([])
                            for j, entry in enumerate(row):
                                lfa_grid = self.get_lfa_grid(entry.grid, i)
                                partition_stencils = expression.partitioning.generate(entry.generate_stencil(), entry.grid)
                                lfa_red = stencil_to_lfa(partition_stencils[0], lfa_grid)
                                lfa_black = stencil_to_lfa(partition_stencils[1], lfa_grid)
                                if i == j:
                                    red_entries[-1].append(lfa_red)
                                    black_entries[-1].append(lfa_black)
                                else:
                                    red_entries[-1].append(lfa_red * lfa_red.matching_zero())
                                    black_entries[-1].append(lfa_black * lfa_black.matching_zero())
                        red_filter = lfa_lab.system(red_entries)
                        black_filter = lfa_lab.system(black_entries)
                        result = (black_filter + red_filter * tmp) * (red_filter + black_filter * tmp)
                    else:
                        raise RuntimeError("Computation could not be partitioned.")
                else:
                    raise RuntimeError("Computation could not be partitioned.")
            else:
                raise NotImplementedError("Not implemented")
        elif isinstance(expression, base.Residual):
            operator = self.transform(expression.operator)
            if isinstance(expression.rhs, system.RightHandSide):
                rhs = operator.matching_zero()
            else:
                rhs = self.transform(expression.rhs)
            if isinstance(expression.approximation, system.ZeroApproximation):
                approximation = rhs.matching_zero()
            elif isinstance(expression.approximation, system.Approximation):
                approximation = rhs.matching_identity()
            else:
                approximation = self.transform(expression.approximation)
            result = rhs - operator * approximation
            # result = self.transform(expression.generate_expression())
        elif isinstance(expression, base.BinaryExpression):
            child1 = self.transform(expression.operand1)
            child2 = self.transform(expression.operand2)
            if isinstance(expression, base.Multiplication):
                result = child1 * child2
            elif isinstance(expression, base.Addition):
                result = child1 + child2
            elif isinstance(expression, base.Subtraction):
                result = child1 - child2
            else:
                raise RuntimeError("Not evaluated")
        elif isinstance(expression, base.Scaling):
            result = expression.factor * self.transform(expression.operand)
        elif isinstance(expression, base.Inverse):
            result = self.transform(expression.operand).inverse()
        elif isinstance(expression, system.Diagonal):
            result = self.transform(expression.operand).diag()
        elif isinstance(expression, system.ElementwiseDiagonal):
            result = self.transform(expression.operand).elementwise_diag()
        elif isinstance(expression, base.CoarseGridSolver):
            operator = self.transform(expression.operator)
            result = operator.inverse()
        elif isinstance(expression, system.Operator):
            lfa_entries = []
            for operator_row in expression.entries:
                lfa_entries.append([])
                for i, entry in enumerate(operator_row):
                    if isinstance(entry, base.InterGridOperator):
                        operator = entry
                        lfa_fine_grid = self.get_lfa_grid(operator.fine_grid, i)
                        lfa_coarse_grid = self.get_lfa_grid(operator.coarse_grid, i)
                        stencil = operator.generate_stencil()
                        lfa_stencil = stencil_to_lfa(stencil, lfa_fine_grid)
                        if isinstance(operator, base.Restriction):
                            lfa_operator = lfa_lab.injection_restriction(lfa_fine_grid, lfa_coarse_grid) * lfa_stencil
                        elif isinstance(operator, base.Prolongation):
                            lfa_operator = lfa_stencil * lfa_lab.injection_interpolation(lfa_fine_grid, lfa_coarse_grid)
                        else:
                            raise NotImplementedError("Not implemented")
                    else:
                        lfa_grid = self.get_lfa_grid(entry.grid, i)
                        lfa_operator = stencil_to_lfa(entry.generate_stencil(), lfa_grid)
                    lfa_entries[-1].append(lfa_operator)
            result = lfa_lab.system(lfa_entries)
        else:
            raise NotImplementedError("Not implemented")
        expression.lfa_symbol = result
        return result

    def compute_spectral_radius(self, expression: base.Expression):
        try:
            lfa_expression = self.transform(expression)

            def evaluate(q, expr):
                try:
                    s = expr.symbol()
                    q.put(s.spectral_radius())

                except (ArithmeticError, RuntimeError, MemoryError) as _:
                    q.put(0.0)

            queue = Queue()
            p = Process(target=evaluate, args=(queue, lfa_expression))
            p.start()
            p.join()
            if queue.empty():
                return 0.0
            return queue.get(timeout=10)
        except (ArithmeticError, RuntimeError, MemoryError) as _:
            return 0.0

    @staticmethod
    def plot_symbol(lfa_operator):
        import matplotlib.pyplot as mpp
        lfa_lab.plot.plot_2d(lfa_operator)
        mpp.show()

    @staticmethod
    def compute_eigenvalues(lfa_operator):
        symbol = lfa_operator.symbol()
        eigenvalues = symbol.eigenvalues()
        return eigenvalues
