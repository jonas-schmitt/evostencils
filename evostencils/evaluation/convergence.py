import lfa_lab
import sympy as sp
import numpy as np
from evostencils.expressions import scalar


class ConvergenceEvaluator:

    def __init__(self, operator):
        self._operator = operator

    @property
    def operator(self):
        return self._operator

    def transform(self, expression: sp.MatrixExpr):
        if isinstance(expression, sp.MatMul):
            result = self.transform(expression.args[0]) * self.transform(expression.args[1])
        elif isinstance(expression, sp.MatAdd):
            result = self.transform(expression.args[0]) * self.transform(expression.args[1])
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
        elif isinstance(expression, sp.Transpose):
            result = self.transform(expression.arg).transpose()
        elif isinstance(expression, sp.Identity):
            result = self.operator.matching_identity()
        elif isinstance(expression, sp.ZeroMatrix):
            result = self.operator.matching_zero()
        elif isinstance(expression, sp.MatrixSymbol):
            result = self.operator
        else:
            result = np.float64(expression.evalf())
        return result
