from evostencils.expressions import multigrid
from evostencils.expressions import base


def propagate_zero(expression: base.Expression) -> base.Expression:
    result = expression.apply(propagate_zero)
    if isinstance(result, multigrid.Correction):
        if isinstance(result.iteration_matrix, base.Zero):
            return result.grid
    if isinstance(result, base.Addition):
        if isinstance(result.operand1, base.Zero):
            return result.operand2
        elif isinstance(result.operand2, base.Zero):
            return result.operand1
    elif isinstance(result, base.Subtraction):
        if isinstance(result.operand1, base.Zero):
            if isinstance(result.operand2, base.Zero):
                return result.operand1
            else:
                return base.Scaling(-1, result.operand2)
        elif isinstance(result.operand2, base.Zero):
            return result.operand1
    elif isinstance(result, base.Multiplication):
        if isinstance(result.operand1, base.Zero) or isinstance(result.operand2, base.Zero):
            return base.Zero(expression.shape)
    elif isinstance(result, base.Scaling):
        if isinstance(result.operand, base.Zero):
            return result.operand
    elif isinstance(result, base.Inverse):
        if isinstance(result.operand, base.Zero):
            return result.operand
    elif isinstance(result, base.Transpose):
        if isinstance(result.operand, base.Zero):
            return base.Zero(expression.shape)
    return result


def fold_intergrid_operations(expression: base.Expression) -> base.Expression:
    result = expression.apply(fold_intergrid_operations)
    if isinstance(result, base.Multiplication):
        child1 = result.operand1
        child2 = result.operand2
        if isinstance(child1, multigrid.Interpolation) and isinstance(child2, multigrid.Restriction):
            stencil = expression.generate_stencil()
            if stencil is None:
                return base.Identity(expression.shape)
            else:
                return base.Identity(expression.shape, stencil.dimension)
    return expression


def remove_identity_operations(expression: base.Expression) -> base.Expression:
    result = expression.apply(remove_identity_operations)
    if isinstance(result, base.Subtraction):
        operand1 = result.operand1
        operand2 = result.operand2
        if isinstance(operand1, base.Identity) and isinstance(operand2, base.Identity):
            return base.Zero(expression.shape)
    elif isinstance(result, base.Multiplication):
        operand1 = result.operand1
        operand2 = result.operand2
        if isinstance(operand1, base.Identity) and base.is_quadratic(operand1):
            return operand2
        elif isinstance(operand2, base.Identity) and base.is_quadratic(operand2):
            return operand1
    elif isinstance(result, base.Inverse) or isinstance(result, base.Transpose):
        operand = result.operand
        if isinstance(operand, base.Identity) and base.is_quadratic(operand):
            return operand
    return result


def substitute_entity(expression: base.Expression, source: base.Entity, destination: base.Entity) -> base.Expression:
    if isinstance(expression, multigrid.Correction):
        iteration_matrix = substitute_entity(expression.iteration_matrix, source, destination)
        grid = substitute_entity(expression.grid, source, destination)
        operator = substitute_entity(expression.operator, source, destination)
        rhs = substitute_entity(expression.rhs, source, destination)
        return multigrid.Correction(iteration_matrix, grid, operator, rhs, expression.weight)
    elif isinstance(expression, base.Entity):
        if expression.name == source.name:
            return destination
        else:
            return expression
    elif isinstance(expression, base.UnaryExpression):
        return type(expression)(substitute_entity(expression.operand, source, destination))
    elif isinstance(expression, base.BinaryExpression):
        child1 = substitute_entity(expression.operand1, source, destination)
        child2 = substitute_entity(expression.operand2, source, destination)
        return type(expression)(child1, child2)
    elif isinstance(expression, base.Scaling):
        return base.Scaling(expression.factor, substitute_entity(expression.operand, source, destination))
    else:
        raise NotImplementedError("Not implemented")


def set_weights(expression: base.Expression, weights: list) -> base.Expression:
    if isinstance(expression, multigrid.Correction):
        if len(weights) == 0:
            raise RuntimeError("Too few weights have been supplied")
        head, *tail = weights
        return multigrid.Correction(expression.iteration_matrix, set_weights(expression.grid, tail), expression.operator, expression.rhs, weight=head)
    elif isinstance(expression, base.Grid):
        if len(weights) > 0:
            raise RuntimeError("Too many weights have been supplied")
        return expression
    else:
        raise NotImplementedError("Not implemented")


def obtain_weights(expression: base.Expression) -> list:
    weights = []
    if isinstance(expression, multigrid.Correction):
        weights.append(expression.weight)
        weights.extend(obtain_weights(expression.grid))
        return weights
    elif isinstance(expression, base.Grid):
        return weights
    else:
        raise NotImplementedError("Not implemented")

