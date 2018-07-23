from evostencils.expressions import multigrid
from evostencils.expressions import base


def propagate_zero(expression):
    if isinstance(expression, base.Addition):
        pass
    elif isinstance(expression, base.Multiplication):
        pass
    return expression


def fold_intergrid_operations(expression):
    if isinstance(expression, base.Multiplication):
        pass
    return expression



