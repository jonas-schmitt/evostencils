from evostencils.expressions import multigrid as mg
from evostencils.expressions import base

# Not bugfree
"""
def propagate_zero(expression: base.Expression) -> base.Expression:
    result = expression.apply(propagate_zero)
    if isinstance(result, multigrid.Cycle):
        if isinstance(result.correction, base.ZeroGrid):
            return result.iterate
    elif isinstance(result, base.Addition):
        if isinstance(result.operand1, base.ZeroOperator):
            return result.operand2
        elif isinstance(result.operand2, base.ZeroOperator):
            return result.operand1
    elif isinstance(result, base.Subtraction):
        if isinstance(result.operand1, base.ZeroOperator):
            if isinstance(result.operand2, base.ZeroOperator):
                return result.operand1
            else:
                return base.Scaling(-1, result.operand2)
        elif isinstance(result.operand2, base.ZeroOperator):
            return result.operand1
    elif isinstance(result, base.Multiplication):
        if isinstance(result.operand1, base.ZeroOperator):
            if isinstance(result.operand2, base.Grid):
                return base.ZeroGrid(result.operand2.size, result.operand2.step_size)
        elif isinstance(result.operand2, base.ZeroOperator):
            return base.ZeroOperator(expression.shape, result.operand2.grid)
    elif isinstance(result, base.Scaling):
        if isinstance(result.operand, base.ZeroOperator):
            return result.operand
    elif isinstance(result, base.Inverse):
        if isinstance(result.operand, base.ZeroOperator):
            return result.operand
    elif isinstance(result, base.Transpose):
        if isinstance(result.operand, base.ZeroOperator):
            return base.ZeroOperator(expression.shape)
    return result


def fold_intergrid_operations(expression: base.Expression) -> base.Expression:
    result = expression.apply(fold_intergrid_operations)
    if isinstance(result, base.Multiplication):
        child1 = result.operand1
        child2 = result.operand2
        if isinstance(child1, multigrid.Interpolation) and isinstance(child2, multigrid.Restriction):
            stencil = result.generate_stencil()
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
            return base.ZeroOperator(expression.shape)
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

def substitute_entity(expression: base.Expression, sources: list, destinations: list, f: lambda x: x) -> base.Expression:
    result = expression.apply(substitute_entity, sources, destinations, f)
    if isinstance(result, base.Entity):
        for source, destination in zip(sources, destinations):
            if result.name == source.name:
                return destination
        return f(result)
    return result
"""


def get_iteration_matrix(expression: base.Expression):
    result = expression.apply(get_iteration_matrix)
    if isinstance(result, mg.Cycle):
        if isinstance(result.iterate, base.ZeroOperator):
            return result.correction
        elif isinstance(result.correction, base.ZeroOperator):
            return result.iterate
        else:
            return result
    elif isinstance(result, mg.Residual):
        operand1 = result.rhs
        if isinstance(result.iterate, base.ZeroOperator):
            operand2 = result.iterate
        else:
            operand2 = base.mul(result.operator, result.iterate)
        # Inefficient but sufficient for now
        if isinstance(operand1, base.ZeroOperator):
            if isinstance(operand2, base.ZeroOperator):
                return operand2
            else:
                return base.Scaling(-1, operand2)
        elif isinstance(operand2, base.ZeroOperator):
            return operand1
        else:
            return base.Subtraction(operand1, operand2)

    elif isinstance(result, base.Addition):
        if isinstance(result.operand1, base.ZeroOperator):
            return result.operand2
        elif isinstance(result.operand2, base.ZeroOperator):
            return result.operand1
        else:
            return result
    elif isinstance(result, base.Subtraction):
        if isinstance(result.operand1, base.ZeroOperator):
            if isinstance(result.operand2, base.ZeroOperator):
                return result.operand2
            else:
                return base.Scaling(-1, result.operand2)
        elif isinstance(result.operand2, base.ZeroOperator):
            return result.operand1
        else:
            return result
    elif isinstance(result, base.Multiplication):
        if isinstance(result.operand1, base.ZeroOperator) or isinstance(result.operand2, base.ZeroOperator):
            return base.ZeroOperator(result.shape, result.grid)
        elif isinstance(result.operand1, base.Identity):
            return result.operand2
        elif isinstance(result.operand2, base.Identity):
            return result.operand1
        else:
            return result
    elif isinstance(result, base.Scaling):
        if isinstance(result.operand, base.ZeroOperator):
            return result.operand
        else:
            return result
    elif isinstance(result, base.ZeroGrid) or isinstance(result, base.RightHandSide):
        return base.ZeroOperator((result.shape[0], result.shape[0]), result)
    elif isinstance(result, base.Grid):
        return base.Identity((result.shape[0], result.shape[0]), result)
    else:
        return result


# Not bugfree
def set_weights(expression: base.Expression, weights: list) -> list:
    if isinstance(expression, mg.Cycle):
        if len(weights) == 0:
            raise RuntimeError("Too few weights have been supplied")
        head, *tail = weights
        expression._weight = head
        return set_weights(expression.correction, tail)
    elif isinstance(expression, mg.Residual):
        tail = set_weights(expression.rhs, weights)
        return set_weights(expression.iterate, tail)
    elif isinstance(expression, base.UnaryExpression) or isinstance(expression, base.Scaling):
        return set_weights(expression.operand, weights)
    elif isinstance(expression, base.BinaryExpression):
        tail = set_weights(expression.operand1, weights)
        return set_weights(expression.operand2, tail)
    else:
        return weights


def obtain_weights(expression: base.Expression) -> list:
    weights = []
    if isinstance(expression, mg.Cycle):
        weights.append(expression.weight)
        weights.extend(obtain_weights(expression.correction))
        return weights
    elif isinstance(expression, mg.Residual):
        weights.extend(obtain_weights(expression.rhs))
        weights.extend(obtain_weights(expression.iterate))
        return weights
    elif isinstance(expression, base.UnaryExpression) or isinstance(expression, base.Scaling):
        weights.extend(obtain_weights(expression.operand))
        return weights
    elif isinstance(expression, base.BinaryExpression):
        weights.extend(obtain_weights(expression.operand1))
        weights.extend(obtain_weights(expression.operand2))
        return weights
    else:
        return weights


def obtain_iterate(expression: base.Expression):
    if isinstance(expression, base.BinaryExpression):
        return obtain_iterate(expression.operand2)
    elif isinstance(expression, base.Grid):
        return expression


def repeat(cycle: mg.Cycle, times):
    def replace_iterate(expression: base.Expression, iterate, new_iterate):
        if isinstance(expression, mg.Residual):
            #if str(expression.iterate) == str(iterate) and expression.iterate.grid.size == iterate.grid.size \
            if expression.iterate.grid.size == iterate.grid.size \
                    and expression.iterate.grid.step_size == iterate.grid.step_size:
                return mg.Residual(expression.operator, new_iterate, expression.rhs)
            else:
                return expression.apply(replace_iterate, iterate, new_iterate)
        else:
            return expression.apply(replace_iterate, iterate, new_iterate)
    new_cycle = cycle
    for _ in range(1, times):
        new_correction = replace_iterate(new_cycle.correction, new_cycle.iterate, new_cycle)
        new_cycle = mg.cycle(new_cycle, new_cycle.rhs, new_correction, partitioning=new_cycle.partitioning,
                             weight=new_cycle.weight, predecessor=new_cycle.predecessor)
    return new_cycle
