import lfa_lab
import evostencils.stencils.periodic as periodic
import evostencils.stencils.constant as constant
from evostencils.expressions import base, multigrid, system
from multiprocessing import Process, Queue
from queue import Empty


@periodic.convert_constant_stencils
def stencil_to_lfa(stencil: periodic.Stencil, grid):
    def recursive_descent(array, dimension):
        if dimension == 1:
            return [lfa_lab.SparseStencil(element.entries) for element in array]
        else:
            return [recursive_descent(element, dimension - 1) for element in array]

    tmp = recursive_descent(stencil.constant_stencils, stencil.dimension)

    ndarray = lfa_lab.NdArray(tmp)
    tmp = lfa_lab.PeriodicStencil(ndarray)
    return lfa_lab.from_periodic_stencil(tmp, grid)


def lfa_sparse_stencil_to_constant_stencil(stencil: lfa_lab.SparseStencil):
    return constant.Stencil(tuple(entry for entry in stencil), stencil.dim)


class ConvergenceEvaluator:

    def __init__(self, finest_grid, coarsening_factor, dimension, lfa_interpolation_generator,
                 lfa_restriction_generator):
        self._finest_grid = finest_grid
        self._coarsening_factor = coarsening_factor
        self._dimension = dimension
        self._lfa_interpolation_generator = lfa_interpolation_generator
        self._lfa_restriction_generator = lfa_restriction_generator

    @property
    def finest_grid(self):
        return self._finest_grid

    def set_finest_grid(self, new_finest_grid):
        self._finest_grid = new_finest_grid

    @property
    def coarsening_factor(self):
        return self._coarsening_factor

    @property
    def dimension(self):
        return self._dimension

    @property
    def lfa_interpolation_generator(self):
        return self._lfa_interpolation_generator

    @property
    def lfa_restriction_generator(self):
        return self._lfa_restriction_generator

    def get_lfa_grid(self, u: base.Grid):
        grid = self.finest_grid
        step_size = grid.step_size()
        while step_size < u.step_size:
            grid = grid.coarse(self.coarsening_factor)
            step_size = grid.step_size()
        return grid

    def transform(self, expression: base.Expression):
        if expression.lfa_symbol is not None:
            return expression.lfa_symbol
        if isinstance(expression, multigrid.Cycle):
            identity = base.Identity(expression.grid)
            tmp = base.Addition(identity, base.Scaling(expression.weight, expression.correction))
            stencil = tmp.generate_stencil()
            partition_stencils = expression.partitioning.generate(stencil, expression.grid)
            if len(partition_stencils) == 1:
                result = self.transform(expression.generate_expression())
            elif len(partition_stencils) == 2:
                u = self.transform(expression.approximation)
                correction = self.transform(expression.correction)
                cycle = u + expression.weight * correction
                lfa_grid = self.get_lfa_grid(expression.correction.grid)
                partition_stencils = [stencil_to_lfa(s, lfa_grid) for s in partition_stencils]
                result = (partition_stencils[0] + partition_stencils[1] * cycle) * \
                         (partition_stencils[1] + partition_stencils[0] * cycle)
            else:
                raise NotImplementedError("Not implemented")
        elif isinstance(expression, base.BinaryExpression):
            child1 = self.transform(expression.operand1)
            child2 = self.transform(expression.operand2)
            if isinstance(expression, base.Multiplication):
                result = child1 * child2
            elif isinstance(expression, base.Addition):
                result = child1 + child2
            elif isinstance(expression, base.Subtraction):
                result = child1 - child2
        elif isinstance(expression, base.Scaling):
            result = expression.factor * self.transform(expression.operand)
        elif isinstance(expression, base.Inverse):
            result = self.transform(expression.operand).inverse()
        elif isinstance(expression, base.Transpose):
            result = self.transform(expression.operand).transpose()
        elif isinstance(expression, base.Diagonal):
            result = self.transform(expression.operand).diag()
        elif isinstance(expression, base.BlockDiagonal):
            stencil = expression.generate_stencil()
            lfa_grid = self.get_lfa_grid(expression.grid)
            result = stencil_to_lfa(stencil, lfa_grid)
        elif isinstance(expression, base.LowerTriangle):
            result = self.transform(expression.operand).lower()
        elif isinstance(expression, base.UpperTriangle):
            result = self.transform(expression.operand).upper()
        elif isinstance(expression, base.Identity):
            lfa_grid = self.get_lfa_grid(expression.grid)
            result = lfa_lab.identity(lfa_grid)
        elif isinstance(expression, base.ZeroOperator):
            lfa_grid = self.get_lfa_grid(expression.grid)
            result = lfa_lab.zero(lfa_grid)
        elif isinstance(expression, multigrid.Restriction):
            lfa_fine_grid = self.get_lfa_grid(expression.fine_grid)
            lfa_coarse_grid = self.get_lfa_grid(expression.coarse_grid)
            result = self.lfa_restriction_generator(lfa_fine_grid, lfa_coarse_grid)
        elif isinstance(expression, multigrid.Prolongation):
            lfa_fine_grid = self.get_lfa_grid(expression.fine_grid)
            lfa_coarse_grid = self.get_lfa_grid(expression.coarse_grid)
            result = self.lfa_interpolation_generator(lfa_fine_grid, lfa_coarse_grid)
        elif isinstance(expression, multigrid.CoarseGridSolver):
            cgs_expression = expression.expression
            if cgs_expression is None or not cgs_expression.evaluate:
                lfa_grid = self.get_lfa_grid(expression.grid)
                stencil = expression.operator.generate_stencil()
                operator = stencil_to_lfa(stencil, lfa_grid)
                result = operator.inverse()
            else:
                if cgs_expression.iteration_matrix is None or cgs_expression.iteration_matrix.lfa_symbol is None:
                    raise RuntimeError("Not evaluated")
                result = cgs_expression.iteration_matrix.lfa_symbol
        elif isinstance(expression, base.Operator):
            lfa_grid = self.get_lfa_grid(expression.grid)
            operator = stencil_to_lfa(expression.generate_stencil(), lfa_grid)
            result = operator
        else:
            raise NotImplementedError("Not implemented")
        expression.lfa_symbol = result
        return result

    def compute_spectral_radius(self, iteration_matrix: base.Expression):

        try:
            lfa_expression = self.transform(iteration_matrix)

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
            return queue.get(timeout=1)
        except (ArithmeticError, RuntimeError, MemoryError) as _:
            return 0.0


class ConvergenceEvaluatorSystem:

    def __init__(self, lfa_grid, coarsening_factors, dimension, lfa_interpolation_generator,
                 lfa_restriction_generator):
        self._lfa_grid = lfa_grid
        self._coarsening_factors = coarsening_factors
        self._dimension = dimension
        self._lfa_interpolation_generator = lfa_interpolation_generator
        self._lfa_restriction_generator = lfa_restriction_generator

    @property
    def lfa_grid(self):
        return self._lfa_grid

    def set_grid(self, new_lfa_grid):
        self._lfa_grid = new_lfa_grid

    @property
    def coarsening_factors(self):
        return self._coarsening_factors

    @property
    def dimension(self):
        return self._dimension

    @property
    def lfa_interpolation_generator(self):
        return self._lfa_interpolation_generator

    @property
    def lfa_restriction_generator(self):
        return self._lfa_restriction_generator

    def get_lfa_grid(self, grid: [base.Grid]):
        lfa_grid = self.lfa_grid
        while all(lfa_subgrid.step_size() < subgrid.step_size for lfa_subgrid, subgrid in zip(lfa_grid, grid)):
            lfa_grid = [lfa_subgrid.coarse(coarsening_factor) for lfa_subgrid, coarsening_factor in zip(lfa_grid, self.coarsening_factors)]
        return lfa_grid

    def transform(self, expression: base.Expression):
        if expression.lfa_symbol is not None:
            return expression.lfa_symbol
        if isinstance(expression, multigrid.Cycle):
            result = self.transform(expression.generate_expression())
        elif isinstance(expression, base.BinaryExpression):
            child1 = self.transform(expression.operand1)
            child2 = self.transform(expression.operand2)
            if isinstance(expression, base.Multiplication):
                result = child1 * child2
            elif isinstance(expression, base.Addition):
                result = child1 + child2
            elif isinstance(expression, base.Subtraction):
                result = child1 - child2
        elif isinstance(expression, base.Scaling):
            result = expression.factor * self.transform(expression.operand)
        elif isinstance(expression, base.Inverse):
            result = self.transform(expression.operand).inverse()
        elif isinstance(expression, multigrid.CoarseGridSolver):
            cgs_expression = expression.expression
            if cgs_expression is None or not cgs_expression.evaluate:
                operator = self.transform(expression.operator)
                result = operator.inverse()
            else:
                if cgs_expression.iteration_matrix is None or cgs_expression.iteration_matrix.lfa_symbol is None:
                    raise RuntimeError("Not evaluated")
                result = cgs_expression.iteration_matrix.lfa_symbol
        elif isinstance(expression, system.Operator):
            lfa_grid = expression.grid
            #TODO implementation
            result = None
        else:
            raise NotImplementedError("Not implemented")
        expression.lfa_symbol = result
        return result

    def compute_spectral_radius(self, iteration_matrix: base.Expression):

        try:
            lfa_expression = self.transform(iteration_matrix)

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
            return queue.get(timeout=1)
        except (ArithmeticError, RuntimeError, MemoryError) as _:
            return 0.0

