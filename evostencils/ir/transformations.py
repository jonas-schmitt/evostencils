from evostencils.ir import base, system
from evostencils.stencils import constant, multiple
import sympy


def obtain_iterate(expression: base.Expression):
    if isinstance(expression, base.BinaryExpression):
        return obtain_iterate(expression.operand2)
    elif isinstance(expression, base.Approximation):
        return expression


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
    if expression is not None:
        expression.lfa_symbol = None
        expression.valid = False
        expression.mutate(invalidate_expression)


def obtain_sympy_expression_for_local_system(smoothing_operator, system_operator, equations, fields):
    local_equations = {}

    def recursive_descent(array1, array2, dimension, index, equation, ii, jj, level_):
        def constant_stencil_to_equation(constant_stencil: constant.Stencil, equation_, new, f, index_):
            for offsets_, value_ in constant_stencil.entries:
                symbol_name = f'{fields[jj].name}@{level_}@['
                for idx, o in zip(index_[:-1], offsets_[:-1]):
                    symbol_name += f'{int(idx) + int(o)}, '
                symbol_name += f'{int(index_[-1]) + int(offsets_[-1])}]'
                if new:
                    symbol_name += '_new'
                equation_ = f(equation_, value_ * sympy.Symbol(symbol_name))
            return equation_

        max_period = max(len(array1), len(array2))
        if dimension == 1:
            for k in range(max_period):
                index_center = index + (k,)
                equation = sympy.sympify(0)
                equation = constant_stencil_to_equation(array1[k % len(array1)], equation, True, lambda x, y: x + y, index_center)
                equation = constant_stencil_to_equation(array1[k % len(array1)], equation, False, lambda x, y: x - y, index_center)
                equation = constant_stencil_to_equation(array2[k % len(array2)], equation, False, lambda x, y: x + y, index_center)
                if (fields[ii], level_, index_center) not in local_equations:
                    field = fields[ii]
                    rhs_name = None
                    for eq_info in equations:
                        if eq_info.level == level_ and eq_info.associated_field == field:
                            rhs_name = eq_info.rhs_name
                    if rhs_name is None:
                        raise RuntimeError("Local solve generation: Could not associate right-hand side with field")
                    local_equations[(fields[ii], level_, index_center)] = sympy.sympify(0), \
                                                                          sympy.Symbol(f'{rhs_name}@{level_}')
                value = local_equations[(fields[ii], level_, index_center)]
                local_equations[(fields[ii], level_, index_center)] = value[0] + equation, value[1]
        else:
            for k in range(max_period):
                recursive_descent(array1[k % len(array1)], array2[k % len(array2)], dimension - 1, index + (k,),
                                  equation, ii, jj, level_)
    if isinstance(smoothing_operator, system.Diagonal):
        for i, (row1, row2) in enumerate(zip(smoothing_operator.operand.entries, system_operator.entries)):
            for j, (entry1, entry2) in enumerate(zip(row1, row2)):
                level = entry2.grid.level
                if i == j:
                    stencil1 = multiple.diagonal(entry1.generate_stencil())
                else:
                    stencil1 = multiple.map_stencil(constant.get_null_stencil(entry1.grid), lambda x: x)
                stencil2 = multiple.map_stencil(entry2.generate_stencil(), lambda x: x)
                recursive_descent(stencil1.constant_stencils, stencil2.constant_stencils, entry2.grid.dimension, (),
                                  sympy.sympify(0), i, j, level)
    elif isinstance(smoothing_operator, system.ElementwiseDiagonal):
        for i, (row1, row2) in enumerate(zip(smoothing_operator.operand.entries, system_operator.entries)):
            for j, (entry1, entry2) in enumerate(zip(row1, row2)):
                level = entry2.grid.level
                stencil1 = multiple.diagonal(entry1.generate_stencil())
                stencil2 = multiple.map_stencil(entry2.generate_stencil(), lambda x: x)
                recursive_descent(stencil1.constant_stencils, stencil2.constant_stencils, entry2.grid.dimension, (),
                                  sympy.sympify(0), i, j, level)
    elif isinstance(smoothing_operator, system.Operator):
        # Custom smoothing operator
        for i, (row1, row2) in enumerate(zip(smoothing_operator.entries, system_operator.entries)):
            for j, (entry1, entry2) in enumerate(zip(row1, row2)):
                level = entry2.grid.level
                stencil1 = multiple.map_stencil(entry1.generate_stencil(), lambda x: x)
                stencil2 = multiple.map_stencil(entry2.generate_stencil(), lambda x: x)
                recursive_descent(stencil1.constant_stencils, stencil2.constant_stencils, entry2.grid.dimension, (),
                                  sympy.sympify(0), i, j, level)
    else:
        raise RuntimeError("Can not extract equations from smoothing operator")

    return local_equations


def find_independent_equation_sets(equations_dict: dict):
    independent_set = []
    dependent_set = []
    items = equations_dict.items()
    for i, (key, value) in enumerate(items):
        free_symbols = value[0].free_symbols
        unknowns = []
        for symbol in free_symbols:
            tokens = symbol.name.split('_')
            if tokens[-1] == 'new':
                unknowns.append(symbol)
        is_independent = True
        for j, (_, v) in enumerate(items):
            if i != j:
                for unknown in unknowns:
                    if unknown in v[0].free_symbols:
                        is_independent = False
        if is_independent:
            independent_set.append((key, value))
        else:
            dependent_set.append((key, value))
    return dependent_set, independent_set
