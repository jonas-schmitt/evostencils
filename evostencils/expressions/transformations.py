from evostencils.expressions import multigrid as mg
from evostencils.expressions import base
from evostencils.expressions import partitioning as part

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
    if expression.iteration_matrix is not None:
        return expression.iteration_matrix
    result = expression.apply(get_iteration_matrix)
    if isinstance(result, mg.Cycle):
        if isinstance(result.iterate, base.ZeroOperator):
            iteration_matrix = base.scale(result.weight, result.correction)
        elif isinstance(result.correction, base.ZeroOperator):
            iteration_matrix = result.iterate
        else:
            iteration_matrix = result
    elif isinstance(result, mg.Residual):
        operand1 = result.rhs
        if isinstance(result.iterate, base.ZeroOperator):
            operand2 = result.iterate
        else:
            operand2 = base.mul(result.operator, result.iterate)
        # Inefficient but sufficient for now
        if isinstance(operand1, base.ZeroOperator):
            if isinstance(operand2, base.ZeroOperator):
                iteration_matrix = operand2
            else:
                iteration_matrix = base.Scaling(-1, operand2)
        elif isinstance(operand2, base.ZeroOperator):
            iteration_matrix = operand1
        else:
            iteration_matrix = base.Subtraction(operand1, operand2)

    elif isinstance(result, base.Addition):
        if isinstance(result.operand1, base.ZeroOperator):
            iteration_matrix = result.operand2
        elif isinstance(result.operand2, base.ZeroOperator):
            iteration_matrix = result.operand1
        else:
            iteration_matrix = result
    elif isinstance(result, base.Subtraction):
        if isinstance(result.operand1, base.ZeroOperator):
            if isinstance(result.operand2, base.ZeroOperator):
                iteration_matrix = result.operand2
            else:
                iteration_matrix = base.Scaling(-1, result.operand2)
        elif isinstance(result.operand2, base.ZeroOperator):
            iteration_matrix = result.operand1
        else:
            iteration_matrix = result
    elif isinstance(result, base.Multiplication):
        if isinstance(result.operand1, base.ZeroOperator) or isinstance(result.operand2, base.ZeroOperator):
            iteration_matrix=  base.ZeroOperator(result.shape, result.grid)
        elif isinstance(result.operand1, base.Identity):
            iteration_matrix = result.operand2
        elif isinstance(result.operand2, base.Identity):
            iteration_matrix = result.operand1
        else:
            iteration_matrix = result
    elif isinstance(result, base.Scaling):
        if isinstance(result.operand, base.ZeroOperator):
            iteration_matrix = result.operand
        else:
            iteration_matrix = result
    elif isinstance(result, base.ZeroGrid) or isinstance(result, base.RightHandSide):
        iteration_matrix = base.ZeroOperator((result.shape[0], result.shape[0]), result)
    elif isinstance(result, base.Grid):
        iteration_matrix = base.Identity((result.shape[0], result.shape[0]), result)
    else:
        iteration_matrix = result
    expression.iteration_matrix = iteration_matrix
    return iteration_matrix


def set_weights(expression: base.Expression, weights: list) -> list:
    if isinstance(expression, mg.Cycle):
        if expression.iteration_matrix is not None:
            expression.iteration_matrix = None
        if isinstance(expression.correction, mg.Residual) \
                or (isinstance(expression.correction, base.Multiplication)
                    and part.can_be_partitioned(expression.correction.operand1)):
            if len(weights) == 0:
                raise RuntimeError("Too few weights have been supplied")
            head, *tail = weights
            expression._weight = head
        else:
            tail = weights
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
        if isinstance(expression.correction, mg.Residual) \
                or (isinstance(expression.correction, base.Multiplication)
                    and part.can_be_partitioned(expression.correction.operand1)):
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


def simplify_iteration_matrix(expression: base.Expression):
    def replace_iterate_with_mutation(expression: base.Expression, iterate, new_iterate):
        if isinstance(expression, mg.Residual):
            if expression.iterate.grid.size == iterate.grid.size \
                    and expression.iterate.grid.step_size == iterate.grid.step_size:
                expression._iterate = new_iterate
            else:
                expression.mutate(replace_iterate_with_mutation, iterate, new_iterate)
        else:
            expression.mutate(replace_iterate_with_mutation, iterate, new_iterate)

    if isinstance(expression, mg.Cycle) and not isinstance(expression.iterate, base.Identity):
        I = base.Identity(expression.iterate.shape, expression.iterate.grid)
        iterate = expression.iterate
        expression._iterate = I
        replace_iterate_with_mutation(expression.correction, iterate, I)
        new_iterate = simplify_iteration_matrix(iterate)
        return base.mul(new_iterate, expression)
    else:
        return expression


def simplify_iteration_matrix_on_all_levels(expression: base.Expression):
    expression.mutate(simplify_iteration_matrix)
    expression.mutate(simplify_iteration_matrix_on_all_levels)


def obtain_coarsest_level(cycle: mg.Cycle) -> int:
    def recursive_descent(expression: base.Expression, current_size: tuple, current_level: int):
        if isinstance(expression, mg.Cycle):
            if expression.grid.size < current_size:
                new_size = expression.grid.size
                new_level = current_level + 1
            else:
                new_size = current_size
                new_level = current_level
            level_iterate = recursive_descent(expression.iterate, new_size, new_level)
            level_correction = recursive_descent(expression.correction, new_size, new_level)
            return max(level_iterate, level_correction)
        elif isinstance(expression, mg.Residual):
            level_iterate = recursive_descent(expression.iterate, current_size, current_level)
            level_rhs = recursive_descent(expression.rhs, current_size, current_level)
            return max(level_iterate, level_rhs)
        elif isinstance(expression, base.BinaryExpression):
            level_operand1 = recursive_descent(expression.operand1, current_size, current_level)
            level_operand2 = recursive_descent(expression.operand2, current_size, current_level)
            return max(level_operand1, level_operand2)
        elif isinstance(expression, base.UnaryExpression):
            return recursive_descent(expression.operand, current_size, current_level)
        elif isinstance(expression, base.Scaling):
            return recursive_descent(expression.operand, current_size, current_level)
        elif isinstance(expression, base.Entity):
            return current_level
        else:
            raise RuntimeError("Unexpected expression")
    return recursive_descent(cycle, cycle.grid.size, 0) + 1


def invalidate_expression(expression: base.Expression):
    def f(expr):
        expr.lfa_symbol = None
        expr.program = None
        expr.iteration_matrix = None
    if expression is not None:
        f(expression)
        expression.mutate(invalidate_expression)
