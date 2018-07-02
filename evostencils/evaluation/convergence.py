import lfa_lab
import sympy


class ConvergenceEvaluator:

    def __init__(self, dimensions, initial_spacing, stencil):
        self._initial_grid = lfa_lab.Grid(dimensions, initial_spacing)
        self._stencil = stencil

    @property
    def initial_grid(self):
        return self._initial_grid

    @property
    def stencil(self):
        return self._stencil

    def evaluate(self, expression: sympy.MatrixExpr):
        pass
