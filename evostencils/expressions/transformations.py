from evostencils.expressions import multigrid
from evostencils.expressions import base


def propagate_zero(expression):
    if isinstance(expression, base.Addition):
        child1 = propagate_zero(expression.operand1)
        child2 = propagate_zero(expression.operand2)
        if isinstance(child1, base.Zero):
            return child2
        elif isinstance(child2, base.Zero):
            return child1
        else:
            return base.Addition(child1, child2)
    elif isinstance(expression, base.Subtraction):
        child1 = propagate_zero(expression.operand1)
        child2 = propagate_zero(expression.operand2)
        if isinstance(child1, base.Zero):
            if isinstance(child2, base.Zero):
                return child1
            else:
                return base.Scaling(-1, child2)
        elif isinstance(child2, base.Zero):
            return child1
    elif isinstance(expression, base.Multiplication):
        child1 = propagate_zero(expression.operand1)
        child2 = propagate_zero(expression.operand2)
        if isinstance(child1, base.Zero) or isinstance(child2, base.Zero):
            return base.Zero(expression.shape)
        else:
            return base.Multiplication(child1, child2)
    elif isinstance(expression, base.Scaling):
        child = propagate_zero(expression.operand)
        if isinstance(child, base.Zero):
            return child
        else:
            return base.Scaling(expression.factor, child)
    elif isinstance(expression, base.Inverse):
        child = propagate_zero(expression.operand)
        if isinstance(child, base.Zero):
            return child
        else:
            return base.Inverse(child)
    elif isinstance(expression, base.Transpose):
        child = propagate_zero(expression.operand)
        if isinstance(child, base.Zero):
            return base.Zero(expression.shape)
        else:
            return base.Transpose(child)
    else:
        return expression


def fold_intergrid_operations(expression):
    if isinstance(expression, base.Multiplication):
        child1 = fold_intergrid_operations(expression.operand1)
        child2 = fold_intergrid_operations(expression.operand2)
        if isinstance(child1, multigrid.Interpolation) and isinstance(child2, multigrid.Restriction):
            return base.Identity(expression.shape)
        else:
            return base.Multiplication(child1, child2)
    elif isinstance(expression, base.Scaling):
        child = fold_intergrid_operations(expression.operand)
        return base.Scaling(expression.factor, child)
    elif isinstance(expression, base.UnaryExpression):
        child = fold_intergrid_operations(expression.operand)
        return type(expression)(child)
    elif isinstance(expression, base.BinaryExpression):
        child1 = fold_intergrid_operations(expression.operand1)
        child2 = fold_intergrid_operations(expression.operand2)
        return type(expression)(child1, child2)
    else:
        return expression



