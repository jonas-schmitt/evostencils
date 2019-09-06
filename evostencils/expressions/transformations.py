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
    local_equations = {}

    def recursive_descent(array1, array2, dimension, index, equation, i, j, level):
        def constant_stencil_to_equation(constant_stencil: constant.Stencil, equation, new, f):
            for offsets, value in constant_stencil.entries:
                symbol_name = f'{fields[j].name}@{level}@['
                for idx, o in zip(index[:-1], offsets[:-1]):
                    symbol_name += f'{int(idx) + int(o)}, '
                symbol_name += f'{int(index[-1]) + int(offsets[-1])}]'
                if new:
                    symbol_name += '_new'
                equation = f(equation, value * sympy.Symbol(symbol_name))
            return equation

        max_period = max(len(array1), len(array2))
        if dimension == 1:
            for k in range(max_period):
                index = index + (k,)
                equation = constant_stencil_to_equation(array1[k % len(array1)], equation, True, lambda x, y: x + y)
                equation = constant_stencil_to_equation(array1[k % len(array1)], equation, False, lambda x, y: x - y)
                equation = constant_stencil_to_equation(array2[k % len(array2)], equation, False, lambda x, y: x + y)
                if (fields[i], index) not in local_equations:
                    local_equations[(fields[i], index)] = sympy.sympify(0), \
                                                          sympy.Symbol(f'{equations[i].rhs_name}@{level}')
                value = local_equations[(fields[i], index)]
                local_equations[(fields[i], index)] = value[0] + equation, value[1]

        else:
            for k in range(max_period):
                recursive_descent(array1[k % len(array1)], array2[k % len(array2)], dimension - 1, index + (k,),
                                  equation, i, j, level)
    if isinstance(smoothing_operator, system.Diagonal):
        for i, (row1, row2) in enumerate(zip(smoothing_operator.operand.entries, system_operator.entries)):
            for j, (entry1, entry2) in enumerate(zip(row1, row2)):
                level = entry2.grid.level
                if i == j:
                    stencil1 = periodic.diagonal(entry1.generate_stencil())
                else:
                    stencil1 = periodic.map_stencil(constant.get_null_stencil(entry1.grid), lambda x: x)
                stencil2 = periodic.map_stencil(entry2.generate_stencil(), lambda x: x)
                recursive_descent(stencil1.constant_stencils, stencil2.constant_stencils, entry2.grid.dimension, (),
                                  sympy.sympify(0), i, j, level)
    elif isinstance(smoothing_operator, system.ElementwiseDiagonal):
        for i, (row1, row2) in enumerate(zip(smoothing_operator.operand.entries, system_operator.entries)):
            for j, (entry1, entry2) in enumerate(zip(row1, row2)):
                level = entry2.grid.level
                stencil1 = periodic.diagonal(entry1.generate_stencil())
                stencil2 = periodic.map_stencil(entry2.generate_stencil(), lambda x: x)
                recursive_descent(stencil1.constant_stencils, stencil2.constant_stencils, entry2.grid.dimension, (),
                                  sympy.sympify(0), i, j, level)
    elif isinstance(smoothing_operator, system.Operator):
        # Custom smoothing operator
        for i, (row1, row2) in enumerate(zip(smoothing_operator.entries, system_operator.entries)):
            for j, (entry1, entry2) in enumerate(zip(row1, row2)):
                level = entry2.grid.level
                stencil1 = periodic.map_stencil(entry1.generate_stencil(), lambda x: x)
                stencil2 = periodic.map_stencil(entry2.generate_stencil(), lambda x: x)
                recursive_descent(stencil1.constant_stencils, stencil2.constant_stencils, entry2.grid.dimension, (),
                                  sympy.sympify(0), i, j, level)
    else:
        raise RuntimeError("Can not extract equations from smoothing operator")

    return local_equations
