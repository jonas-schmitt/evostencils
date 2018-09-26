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

    def __init__(self, grid, coarsening_factor, dimension, operator, interpolation, restriction):
        self._grid = grid
        self._coarsening_factor = coarsening_factor
        self._dimension = dimension
        self._operator = operator
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
    def operator(self):
        return self._operator

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def restriction(self):
        return self._restriction

    def get_grid_on_level(self, level):
        grid = self.grid
        for _ in range(level):
            grid = grid.coarse(self.coarsening_factor)
        return grid

    def get_operator_on_grid(self, grid):
        return self.operator(grid)

    def transform(self, expression: base.Expression, level):
        if isinstance(expression, multigrid.Cycle):
            identity = base.Identity((expression.grid.shape[0], expression.grid.shape[0]), self.dimension)
            tmp = base.Addition(identity, base.Scaling(expression.weight, expression.correction))
            stencil = tmp.generate_stencil()
            partition_stencils = expression.partitioning.generate(stencil)
            if len(partition_stencils) == 1:
                return self.transform(expression.generate_expression(), level)
            elif len(partition_stencils) == 2:
                u = self.transform(expression.grid, level)
                correction = self.transform(expression.correction, level)
                cycle = u + expression.weight * correction
                grid = self.get_grid_on_level(level)
                partition_stencils = [stencil_to_lfa(s, grid) for s in partition_stencils]
                return (partition_stencils[0] + partition_stencils[1] * cycle) \
                    * (partition_stencils[1] + partition_stencils[0] * cycle)
            else:
                raise NotImplementedError("Not implemented")
        elif isinstance(expression, base.BinaryExpression):
            if isinstance(expression, base.Multiplication):
                if isinstance(expression.operand1, multigrid.CoarseGridSolver) or isinstance(expression.operand1, multigrid.Cycle):
                    child1 = self.transform(expression.operand1, level+1)
                else:
                    child1 = self.transform(expression.operand1, level)
                if isinstance(expression.operand2, multigrid.CoarseGridSolver) or isinstance(expression.operand2, multigrid.Cycle):
                    child2 = self.transform(expression.operand2, level+1)
                else:
                    child2 = self.transform(expression.operand2, level)
                return child1 * child2
            elif isinstance(expression, base.Addition):
                child1 = self.transform(expression.operand1, level)
                child2 = self.transform(expression.operand2, level)
                return child1 + child2
            elif isinstance(expression, base.Subtraction):
                child1 = self.transform(expression.operand1, level)
                child2 = self.transform(expression.operand2, level)
                return child1 - child2
        elif isinstance(expression, base.Scaling):
            return expression.factor * self.transform(expression.operand, level)
        elif isinstance(expression, base.Inverse):
            return self.transform(expression.operand, level).inverse()
        elif isinstance(expression, base.Transpose):
            return self.transform(expression.operand, level).transpose()
        elif isinstance(expression, base.Diagonal):
            return self.transform(expression.operand, level).diag()
        elif isinstance(expression, base.BlockDiagonal):
            grid = self.get_grid_on_level(level)
            stencil = expression.generate_stencil()
            return stencil_to_lfa(stencil, grid)
        elif isinstance(expression, base.LowerTriangle):
            return self.transform(expression.operand, level).lower()
        elif isinstance(expression, base.UpperTriangle):
            return self.transform(expression.operand, level).upper()
        elif isinstance(expression, base.Identity):
            grid = self.get_grid_on_level(level)
            return lfa_lab.identity(grid)
        elif isinstance(expression, base.ZeroOperator):
            grid = self.get_grid_on_level(level)
            return lfa_lab.zero(grid)
        elif type(expression) == multigrid.Restriction:
            grid = self.get_grid_on_level(level)
            coarse_grid = grid.coarse(self.coarsening_factor)
            return self.restriction(grid, coarse_grid)
        elif type(expression) == multigrid.Interpolation:
            grid = self.get_grid_on_level(level)
            coarse_grid = grid.coarse(self.coarsening_factor)
            return self.interpolation(grid, coarse_grid)
        elif type(expression) == multigrid.CoarseGridSolver and isinstance(expression, multigrid.CoarseGridSolver):
            grid = self.get_grid_on_level(level)
            operator = self.get_operator_on_grid(grid)
            return operator.inverse()
        elif isinstance(expression, base.Operator):
            grid = self.get_grid_on_level(level)
            operator = self.get_operator_on_grid(grid)
            return operator
        raise NotImplementedError("Not implemented")

    def compute_spectral_radius(self, expression: base.Expression):
        try:
            smoother = self.transform(expression, 0)
            symbol = smoother.symbol()
            return symbol.spectral_radius()
        except RuntimeError as _:
            return 0.0


