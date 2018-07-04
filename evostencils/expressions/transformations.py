import sympy as sp


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
