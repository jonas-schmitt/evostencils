from evostencils.expressions import multigrid as mg
from evostencils.expressions import base
from evostencils.expressions import system

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


def get_system_iteration_matrix(expression: base.Expression):
    if expression.iteration_matrix is not None:
        return expression.iteration_matrix
    result = expression.apply(get_iteration_matrix)
    if isinstance(result, mg.Cycle):
        if isinstance(result.approximation, system.ZeroOperator):
            iteration_matrix = base.scale(result.weight, result.correction)
        elif isinstance(result.correction, system.ZeroOperator):
            iteration_matrix = result.approximation
        else:
            iteration_matrix = result
    elif isinstance(result, mg.Residual):
        operand1 = result.rhs
        if isinstance(result.approximation, system.ZeroOperator):
            operand2 = result.approximation
        else:
            operand2 = base.mul(result.operator, result.approximation)
        # Inefficient but sufficient for now
        if isinstance(operand1, system.ZeroOperator):
            if isinstance(operand2, system.ZeroOperator):
                iteration_matrix = operand2
            else:
                iteration_matrix = base.Scaling(-1, operand2)
        elif isinstance(operand2, system.ZeroOperator):
            iteration_matrix = operand1
        else:
            iteration_matrix = base.Subtraction(operand1, operand2)
    elif isinstance(result, base.Addition):
        if isinstance(result.operand1, system.ZeroOperator):
            iteration_matrix = result.operand2
        elif isinstance(result.operand2, system.ZeroOperator):
            iteration_matrix = result.operand1
        else:
            iteration_matrix = result
    elif isinstance(result, base.Subtraction):
        if isinstance(result.operand1, system.ZeroOperator):
            if isinstance(result.operand2, system.ZeroOperator):
                iteration_matrix = result.operand2
            else:
                iteration_matrix = base.Scaling(-1, result.operand2)
        elif isinstance(result.operand2, system.ZeroOperator):
            iteration_matrix = result.operand1
        else:
            iteration_matrix = result
    elif isinstance(result, base.Multiplication):
        if isinstance(result.operand1, system.ZeroOperator) or isinstance(result.operand2, system.ZeroOperator):
            iteration_matrix = system.ZeroOperator(result.grid)
        elif isinstance(result.operand1, system.Identity):
            iteration_matrix = result.operand2
        elif isinstance(result.operand2, system.Identity):
            iteration_matrix = result.operand1
        else:
            iteration_matrix = result
    elif isinstance(result, base.Scaling):
        if isinstance(result.operand, system.ZeroOperator):
            iteration_matrix = result.operand
        else:
            iteration_matrix = result
    elif isinstance(result, system.ZeroApproximation) or isinstance(result, system.RightHandSide):
        iteration_matrix = system.ZeroOperator('Z', result.grid)
    elif isinstance(result, system.Approximation):
        iteration_matrix = system.Identity('I', result.grid)
    else:
        iteration_matrix = result
    expression.iteration_matrix = iteration_matrix
    return iteration_matrix


def get_iteration_matrix(expression: base.Expression):
    if expression.iteration_matrix is not None:
        return expression.iteration_matrix
    result = expression.apply(get_iteration_matrix)
    if isinstance(result, mg.Cycle):
        if isinstance(result.approximation, base.ZeroOperator):
            iteration_matrix = base.scale(result.weight, result.correction)
        elif isinstance(result.correction, base.ZeroOperator):
            iteration_matrix = result.approximation
        else:
            iteration_matrix = result
    elif isinstance(result, mg.Residual):
        operand1 = result.rhs
        if isinstance(result.approximation, base.ZeroOperator):
            operand2 = result.approximation
        else:
            operand2 = base.mul(result.operator, result.approximation)
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
            iteration_matrix = base.ZeroOperator(result.grid)
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
    elif isinstance(result, base.ZeroApproximation) or isinstance(result, base.RightHandSide):
        iteration_matrix = base.ZeroOperator(result.grid)
    elif isinstance(result, base.Approximation):
        iteration_matrix = base.Identity(result.grid)
    else:
        iteration_matrix = result
    expression.iteration_matrix = iteration_matrix
    return iteration_matrix


def obtain_iterate(expression: base.Expression):
    if isinstance(expression, base.BinaryExpression):
        return obtain_iterate(expression.operand2)
    elif isinstance(expression, base.Approximation):
        return expression


def repeat(cycle: mg.Cycle, times):
    def replace_iterate(expression: base.Expression, iterate, new_iterate):
        if isinstance(expression, mg.Residual):
            if expression.approximation.approximation.size == iterate.approximation.size \
                    and expression.approximation.approximation.step_size == iterate.approximation.step_size:
                return mg.Residual(expression.operator, new_iterate, expression.rhs)
            else:
                return expression.apply(replace_iterate, iterate, new_iterate)
        else:
            return expression.apply(replace_iterate, iterate, new_iterate)
    new_cycle = cycle
    for _ in range(1, times):
        new_correction = replace_iterate(new_cycle.correction, new_cycle.approximation, new_cycle)
        new_cycle = mg.cycle(new_cycle, new_cycle.rhs, new_correction, partitioning=new_cycle.partitioning,
                             weight=new_cycle.weight, predecessor=new_cycle.predecessor)
    return new_cycle


def obtain_coarsest_level(cycle: mg.Cycle) -> int:
    def recursive_descent(expression: base.Expression, current_size: tuple, current_level: int):
        if isinstance(expression, mg.Cycle):
            if expression.grid.size < current_size:
                new_size = expression.grid.size
                new_level = current_level + 1
            else:
                new_size = current_size
                new_level = current_level
            level_iterate = recursive_descent(expression.approximation, new_size, new_level)
            level_correction = recursive_descent(expression.correction, new_size, new_level)
            return max(level_iterate, level_correction)
        elif isinstance(expression, mg.Residual):
            level_iterate = recursive_descent(expression.approximation, current_size, current_level)
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
