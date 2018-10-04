import lfa_lab
import evostencils.stencils.periodic as periodic
from evostencils.expressions import base, multigrid, transformations


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


class ConvergenceEvaluator:

    def __init__(self, finest_grid, coarsening_factor, dimension, lfa_operator_generator, lfa_interpolation_generator,
                 lfa_restriction_generator):
        self._finest_grid = finest_grid
        self._coarsening_factor = coarsening_factor
        self._dimension = dimension
        self._lfa_operator_generator = lfa_operator_generator
        self._lfa_interpolation_generator = lfa_interpolation_generator
        self._lfa_restriction_generator = lfa_restriction_generator

    @property
    def finest_grid(self):
        return self._finest_grid

    @property
    def coarsening_factor(self):
        return self._coarsening_factor

    @property
    def dimension(self):
        return self._dimension

    @property
    def lfa_operator_generator(self):
        return self._lfa_operator_generator

    @property
    def lfa_interpolation_generator(self):
        return self._lfa_interpolation_generator

    @property
    def lfa_restriction_generator(self):
        return self._lfa_restriction_generator

    def get_lfa_grid(self, u: base.Grid):
        grid = self.finest_grid
        while grid.step_size() < u.step_size:
            grid = grid.coarse(self.coarsening_factor)
        return grid

    def get_lfa_operator_on_grid(self, grid):
        return self.lfa_operator_generator(grid)

    def transform(self, expression: base.Expression):
        if isinstance(expression, multigrid.Cycle):
            identity = base.Identity(expression.iterate.shape, expression.grid)
            tmp = base.Addition(identity, base.Scaling(expression.weight, expression.correction))
            stencil = tmp.generate_stencil()
            partition_stencils = expression.partitioning.generate(stencil, expression.grid)
            if len(partition_stencils) == 1:
                return self.transform(transformations.propagate_zero(expression.generate_expression()))
            elif len(partition_stencils) == 2:
                u = self.transform(expression.iterate)
                correction = self.transform(expression.correction)
                cycle = u + expression.weight * correction
                lfa_grid = self.get_lfa_grid(expression.grid)
                partition_stencils = [stencil_to_lfa(s, lfa_grid) for s in partition_stencils]
                return (partition_stencils[0] + partition_stencils[1] * cycle) \
                    * (partition_stencils[1] + partition_stencils[0] * cycle)
            else:
                raise NotImplementedError("Not implemented")
        elif isinstance(expression, base.BinaryExpression):
            if isinstance(expression, base.Multiplication):
                child1 = self.transform(expression.operand1)
                child2 = self.transform(expression.operand2)
                return child1 * child2
            elif isinstance(expression, base.Addition):
                child1 = self.transform(expression.operand1)
                child2 = self.transform(expression.operand2)
                return child1 + child2
            elif isinstance(expression, base.Subtraction):
                child1 = self.transform(expression.operand1)
                child2 = self.transform(expression.operand2)
                return child1 - child2
        elif isinstance(expression, base.Scaling):
            return expression.factor * self.transform(expression.operand)
        elif isinstance(expression, base.Inverse):
            return self.transform(expression.operand).inverse()
        elif isinstance(expression, base.Transpose):
            return self.transform(expression.operand).transpose()
        elif isinstance(expression, base.Diagonal):
            return self.transform(expression.operand).diag()
        elif isinstance(expression, base.BlockDiagonal):
            stencil = expression.generate_stencil()
            lfa_grid = self.get_lfa_grid(expression.grid)
            return stencil_to_lfa(stencil, lfa_grid)
        elif isinstance(expression, base.LowerTriangle):
            return self.transform(expression.operand).lower()
        elif isinstance(expression, base.UpperTriangle):
            return self.transform(expression.operand).upper()
        elif isinstance(expression, base.Identity):
            lfa_grid = self.get_lfa_grid(expression.grid)
            return lfa_lab.identity(lfa_grid)
        elif isinstance(expression, base.ZeroOperator):
            lfa_grid = self.get_lfa_grid(expression.grid)
            return lfa_lab.zero(lfa_grid)
        elif type(expression) == multigrid.Restriction:
            lfa_fine_grid = self.get_lfa_grid(expression.fine_grid)
            lfa_coarse_grid = self.get_lfa_grid(expression.coarse_grid)
            return self.lfa_restriction_generator(lfa_fine_grid, lfa_coarse_grid)
        elif type(expression) == multigrid.Interpolation:
            lfa_fine_grid = self.get_lfa_grid(expression.fine_grid)
            lfa_coarse_grid = self.get_lfa_grid(expression.coarse_grid)
            return self.lfa_interpolation_generator(lfa_fine_grid, lfa_coarse_grid)
        elif type(expression) == multigrid.CoarseGridSolver:
            lfa_grid = self.get_lfa_grid(expression.grid)
            stencil = expression.operator.generate_stencil()
            operator = stencil_to_lfa(stencil, lfa_grid)
            return operator.inverse()
        elif isinstance(expression, base.Operator):
            lfa_grid = self.get_lfa_grid(expression.grid)
            operator = stencil_to_lfa(expression.generate_stencil(), lfa_grid)
            return operator
        raise NotImplementedError("Not implemented")

    def compute_spectral_radius(self, expression: base.Expression):
        #try:
            smoother = self.transform(expression)
            symbol = smoother.symbol()
            return symbol.spectral_radius()
        #except RuntimeError as _:
        #    return 0.0


