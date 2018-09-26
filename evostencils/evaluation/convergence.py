import lfa_lab
import evostencils.stencils.periodic as periodic
from evostencils.expressions import base, multigrid


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

    def __init__(self, grid, coarsening_factor, dimension, interpolation, restriction):
        self._grid = grid
        self._coarsening_factor = coarsening_factor
        self._dimension = dimension
        self._interpolation = interpolation
        self._restriction = restriction

    @property
    def grid(self):
        return self._grid

    @property
    def coarsening_factor(self):
        return self._coarsening_factor

    @property
    def dimension(self):
        return self._dimension

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def restriction(self):
        return self._restriction

    def transform(self, expression: base.Expression, grid):
        if isinstance(expression, multigrid.Cycle):
            identity = base.Identity((expression.grid.shape[0], expression.grid.shape[0]), self.dimension)
            tmp = base.Addition(identity, base.Scaling(expression.weight, expression.correction))
            stencil = tmp.generate_stencil()
            partition_stencils = expression.partitioning.generate(stencil)
            if len(partition_stencils) == 1:
                return self.transform(expression.generate_expression(), grid)
            elif len(partition_stencils) == 2:
                u = self.transform(expression.grid, grid)
                correction = self.transform(expression.correction, grid)
                cycle = u + expression.weight * correction
                partition_stencils = [stencil_to_lfa(s, self._grid) for s in partition_stencils]
                return (partition_stencils[0] + partition_stencils[1] * cycle) \
                    * (partition_stencils[1] + partition_stencils[0] * cycle)
            else:
                raise NotImplementedError("Not implemented")
        elif isinstance(expression, base.BinaryExpression):
            if isinstance(expression, base.Multiplication):
                if isinstance(expression.operand1, multigrid.Interpolation):
                    child2 = self.transform(expression.operand2, grid.coarse(self.coarsening_factor))
                else:
                    child2 = self.transform(expression.operand2, grid)
                if isinstance(expression.operand2, multigrid.Restriction):
                    child1 = self.transform(expression.operand1, grid.coarse(self.coarsening_factor))
                else:
                    child1 = self.transform(expression.operand1, grid)
                return child1 * child2
            elif isinstance(expression, base.Addition):
                child1 = self.transform(expression.operand1, grid)
                child2 = self.transform(expression.operand2, grid)
                return child1 + child2
            elif isinstance(expression, base.Subtraction):
                child1 = self.transform(expression.operand1, grid)
                child2 = self.transform(expression.operand2, grid)
                return child1 - child2
        elif isinstance(expression, base.Scaling):
            return expression.factor * self.transform(expression.operand, grid)
        elif isinstance(expression, base.Inverse):
            return self.transform(expression.operand, grid).inverse()
        elif isinstance(expression, base.Transpose):
            return self.transform(expression.operand, grid).transpose()
        elif isinstance(expression, base.Diagonal):
            return self.transform(expression.operand, grid).diag()
        elif isinstance(expression, base.BlockDiagonal):
            stencil = expression.generate_stencil()
            return stencil_to_lfa(stencil, grid)
        elif isinstance(expression, base.LowerTriangle):
            return self.transform(expression.operand, grid).lower()
        elif isinstance(expression, base.UpperTriangle):
            return self.transform(expression.operand, grid).upper()
        elif isinstance(expression, base.Identity):
            return lfa_lab.identity(grid)
        elif isinstance(expression, base.ZeroOperator):
            return lfa_lab.zero(grid)
        elif type(expression) == multigrid.Restriction:
            coarse_grid = grid.coarse(self.coarsening_factor)
            return self.restriction(grid, coarse_grid)
        elif type(expression) == multigrid.Interpolation:
            coarse_grid = grid.coarse(self.coarsening_factor)
            return self.interpolation(grid, coarse_grid)
        elif type(expression) == multigrid.CoarseGridSolver and isinstance(expression, multigrid.CoarseGridSolver):
            stencil = expression.operator.generate_stencil()
            return stencil_to_lfa(stencil, grid).inverse()
        elif isinstance(expression, base.Operator):
            stencil = expression.generate_stencil()
            return stencil_to_lfa(stencil, grid)
        raise NotImplementedError("Not implemented")

    def compute_spectral_radius(self, expression: base.Expression):
        try:
            smoother = self.transform(expression)
            symbol = smoother.symbol()
            return symbol.spectral_radius()
        except RuntimeError as _:
            return 0.0


