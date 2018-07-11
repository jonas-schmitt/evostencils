import sympy as sp
from evostencils.expressions import multigrid


def propagate_zero(expression):
    if isinstance(expression, sp.MatAdd):
        acc = propagate_zero(expression.args[0])
        for i in range(1, len(expression.args)):
            child = propagate_zero(expression.args[i])
            if not isinstance(child, sp.ZeroMatrix):
                acc = acc + child
        return acc
    elif isinstance(expression, sp.MatMul):
        acc = propagate_zero(expression.args[0])
        for i in range(1, len(expression.args)):
            child = propagate_zero(expression.args[i])
            if isinstance(child, sp.ZeroMatrix):
                return sp.ZeroMatrix(*expression.shape)
            else:
                acc = acc * child
        return acc
    return expression


def fold_intergrid_operations(expression):
    if isinstance(expression, sp.MatMul):
        begin = 0
        for i in range(0, len(expression.args), 2):
            if not isinstance(expression.args[i], multigrid.Interpolation) or not isinstance(expression.args[i+1], multigrid.Restriction):
                begin = i+1
                acc = fold_intergrid_operations(expression.args[i])
                break
        i = begin
        while i < len(expression.args) - 1:
            if not isinstance(expression.args[i], multigrid.Interpolation) or not isinstance(expression.args[i+1], multigrid.Restriction):
                acc = acc * fold_intergrid_operations(expression.args[i])
                i = i + 1
            else:
                i = i + 2

        for j in range(i, len(expression.args)):
            acc = acc * fold_intergrid_operations(expression.args[j])
        return acc
    elif len(expression.args) > 0 and isinstance(expression.args[0], sp.MatrixExpr):
        new_args = tuple(fold_intergrid_operations(arg) for arg in expression.args)
        return expression.func(*new_args)
    else:
        return expression



