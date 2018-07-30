import lfa_lab
from evostencils.expressions import base, multigrid
from functools import reduce
import operator


class ConvergenceEvaluator:

    def __init__(self, fine_operator, coarse_operator, fine_grid, fine_grid_size, coarsening_factor):
        assert len(fine_grid_size) == len(coarsening_factor), 'Dimensions of the fine grid size and the coarsening factor must match'
        self._fine_operator = fine_operator
        self._fine_grid = fine_grid
        self._fine_grid_size = fine_grid_size
        coarse_grid = fine_grid.coarse(coarsening_factor)
        self._coarse_grid = coarse_grid
        self._coarse_grid_size = (fine_grid_size[i] / coarsening_factor[i] for i in range(0, len(fine_grid_size)))
        self._restriction = lfa_lab.gallery.fw_restriction(fine_grid, coarse_grid)
        self._interpolation = lfa_lab.gallery.ml_interpolation(fine_grid, coarse_grid)
        self._coarse_operator = coarse_operator

    @property
    def fine_operator(self):
        return self._fine_operator

    @property
    def coarse_operator(self):
        return self._coarse_operator

    @property
    def fine_grid_size(self):
        return self._fine_grid_size

    @property
    def coarse_grid_size(self):
        return self._coarse_grid_size

    @property
    def restriction(self):
        return self._restriction

    @property
    def interpolation(self):
        return self._interpolation

    def transform(self, expression: base.Expression):
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
            child = self.transform(expression.operand)
            return expression.factor * child
        elif isinstance(expression, base.Inverse):
            return self.transform(expression.operand).inverse()
        elif isinstance(expression, base.Diagonal):
            result = self.transform(expression.operand).diag()
        elif isinstance(expression, base.LowerTriangle):
            result = self.transform(expression.operand).lower()
        elif isinstance(expression, base.UpperTriangle):
            result = self.transform(expression.operand).upper()
        elif isinstance(expression, base.Identity):
            result = self.fine_operator.matching_identity()
        elif isinstance(expression, base.Zero):
            result = self.fine_operator.matching_zero()
        elif type(expression) == multigrid.Restriction:
            result = self._restriction
        elif type(expression) == multigrid.Interpolation:
            result = self._interpolation
        elif isinstance(expression, sp.MatrixSymbol):
            n = reduce(operator.mul, self.fine_grid_size, 1)
            if expression.shape == (n, n):
                result = self.fine_operator
            else:
                result = self.coarse_operator
        else:
            tmp = expression.evalf()
            result = complex(tmp)
        return result

    def compute_spectral_radius(self, expression: sp.MatrixExpr):
        try:
            smoother = self.transform(expression)
            symbol = smoother.symbol()
            return symbol.spectral_radius()
        except RuntimeError as re:
            return 0.0


