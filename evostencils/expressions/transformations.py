from evostencils.expressions import base, system
from evostencils.stencils import constant, periodic
import sympy

def get_iteration_matrix(expression: base.Expression):
    if expression.iteration_matrix is not None:
        return expression.iteration_matrix
    result = expression.apply(get_iteration_matrix)
    if isinstance(result, base.Cycle):
        if isinstance(result.approximation, base.ZeroOperator):
            iteration_matrix = base.scale(result.weight, result.correction)
        elif isinstance(result.correction, base.ZeroOperator):
            iteration_matrix = result.approximation
        else:
            iteration_matrix = result
    elif isinstance(result, base.Residual):
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


def repeat(cycle: base.Cycle, times):
    def replace_iterate(expression: base.Expression, iterate, new_iterate):
        if isinstance(expression, base.Residual):
            if expression.approximation.approximation.size == iterate.approximation.size \
                    and expression.approximation.approximation.step_size == iterate.approximation.step_size:
                return base.Residual(expression.operator, new_iterate, expression.rhs)
            else:
                return expression.apply(replace_iterate, iterate, new_iterate)
        else:
            return expression.apply(replace_iterate, iterate, new_iterate)
    new_cycle = cycle
    for _ in range(1, times):
        new_correction = replace_iterate(new_cycle.correction, new_cycle.approximation, new_cycle)
        new_cycle = base.cycle(new_cycle, new_cycle.rhs, new_correction, partitioning=new_cycle.partitioning,
                                                       weight=new_cycle.weight, predecessor=new_cycle.predecessor)
    return new_cycle


def obtain_coarsest_level(cycle: base.Cycle) -> int:
    def recursive_descent(expression: base.Expression, current_size: tuple, current_level: int):
        if isinstance(expression, base.Cycle):
            if expression.grid.size < current_size:
                new_size = expression.grid.size
                new_level = current_level + 1
            else:
                new_size = current_size
                new_level = current_level
            level_iterate = recursive_descent(expression.approximation, new_size, new_level)
            level_correction = recursive_descent(expression.correction, new_size, new_level)
            return max(level_iterate, level_correction)
        elif isinstance(expression, base.Residual):
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


def obtain_sympy_expression_for_local_system(smoothing_operator, system_operator, equations, fields):
    if isinstance(smoothing_operator, system.Diagonal):
        pass
    elif isinstance(smoothing_operator, system.ElementwiseDiagonal):
        local_equations = []
        for i, row in enumerate(smoothing_operator.operand.entries):
            local_equation = sympy.sympify(0)
            for j, entry in enumerate(row):
                stencil = entry.generate_stencil()
                field_symbol = fields[j]

                def recursive_descent(array1, array2, dimension, equation):
                    def constant_stencil_to_equation(constant_stencil: constant.Stencil, equation, new, f):
                        for offsets, value in constant_stencil.entries:
                            symbol_name = f'{field_symbol.name}'
                            for offset in offsets:
                                symbol_name += f'_{int(offset)}'
                            if new:
                                symbol_name += '_new'
                            equation = f(equation, value * sympy.Symbol(symbol_name))
                        return equation

                    max_period = max(len(array1), len(array2))
                    if dimension == 1:
                        for k in range(max_period):
                            equation = constant_stencil_to_equation(array1[k % len(array1)], equation, True, lambda x, y: x+y)
                            equation = constant_stencil_to_equation(array1[k % len(array1)], equation, False, lambda x, y: x-y)
                            equation = constant_stencil_to_equation(array2[k % len(array2)], equation, False, lambda x, y: x+y)
                            return equation
                    else:
                        for k in range(max_period):
                            equation += recursive_descent(array1[k % len(array1)], array2[k % len(array2)], dimension - 1, equation)
                        return equation

                stencil1 = periodic.diagonal(stencil)
                stencil2 = periodic.map_stencil(stencil, lambda x: x)
                tmp = recursive_descent(stencil1.constant_stencils, stencil2.constant_stencils, stencil.dimension, sympy.sympify(0))
                local_equation = local_equation + tmp
            local_equations.append(sympy.Eq(local_equation, sympy.Symbol(equations[i].rhs_name)))
        return local_equations
    else:
        # Custom smoothing operator
        pass
