import lfa_lab
import sympy as sp
import numpy as np
from evostencils.expressions import scalar, multigrid
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

    def transform(self, expression: sp.MatrixExpr):
        if isinstance(expression, sp.MatMul):
            acc = self.transform(expression.args[0])
            for i in range(1, len(expression.args)):
                child = self.transform(expression.args[i])
                if isinstance(child, complex):
                    acc = child * acc
                else:
                    acc = acc * child
            result = acc
        elif isinstance(expression, sp.MatAdd):
            acc = self.transform(expression.args[0])
            for i in range(1, len(expression.args)):
                child = self.transform(expression.args[i])
                acc = acc + child
            result = acc
        elif isinstance(expression, sp.Inverse):
            if isinstance(expression.arg, sp.ZeroMatrix):
                result = self.transform(expression.arg)
            else:
                result = self.transform(expression.arg).inverse()
        elif isinstance(expression, scalar.Diagonal):
            result = self.transform(expression.arg).diag()
        elif isinstance(expression, scalar.Lower):
            result = self.transform(expression.arg).lower()
        elif isinstance(expression, scalar.Upper):
            result = self.transform(expression.arg).upper()
        elif isinstance(expression, sp.Identity):
            result = self.fine_operator.matching_identity()
        elif isinstance(expression, sp.ZeroMatrix):
            result = self.fine_operator.matching_zero()
        elif type(expression) == multigrid.Restriction:
            result = self._restriction
        elif type(expression) == multigrid.Interpolation:
            result = self._interpolation
        elif isinstance(expression, sp.MatrixSymbol):
            #TODO dirty fix here to return the right symbol
            #TODO We need a better solution here!
            if expression.shape[0] > expression.shape[1]:
                result = self._interpolation
            elif expression.shape[0] < expression.shape[1]:
                result = self._restriction
            else:
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


