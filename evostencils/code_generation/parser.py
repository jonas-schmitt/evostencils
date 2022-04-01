from sympy.parsing.sympy_parser import parse_expr
from evostencils.ir import base
import re
from evostencils.stencils import constant
from evostencils.grammar import multigrid as initialization


def parse_stencil_offsets(string: str, is_prolongation=False):
    tokens = re.split(r" ?\[ ?| ?\] ?| ?, ?", string)
    tokens = [token for token in tokens if not token == ""]
    offsets = []
    for token in tokens:
        sympy_expr = parse_expr(token)
        substituted_expr = sympy_expr
        for symbol in sympy_expr.free_symbols:
            substituted_expr = substituted_expr.subs(symbol, 0)
        if is_prolongation:
            offset = int(round(2 * substituted_expr.evalf()))
        else:
            offset = int(round(substituted_expr.evalf()))
        offsets.append(offset)
    return tuple(offsets)


def extract_l2_information(file_path: str, dimension: int, solution_equations=None):
    equations = []
    operators = []
    fields = []
    with open(file_path, 'r') as file:
        line = file.readline()
        while line:
            tokens = line.split(' ')
            if tokens[0] == 'Operator':
                stencil_entries = []
                tmp = tokens[1].split('@')
                name = tmp[0]
                level = int(tmp[1])
                line = file.readline()
                tokens = line.split('from')
                is_prolongation = False
                is_restriction = False
                if 'restriction' in name.lower():
                    is_restriction = True
                elif 'prolongation' in name.lower():
                    is_prolongation = True
                while True:
                    tmp = tokens[1].split('with')
                    value = parse_expr(tmp[1])
                    # value = value.evalf()
                    offsets = parse_stencil_offsets(tmp[0], is_prolongation)
                    stencil_entries.append((offsets, value))
                    line = file.readline()
                    tokens = line.split('from')
                    if '}' in line:
                        break
                if is_restriction:
                    operator_type = base.Restriction
                elif is_prolongation:
                    operator_type = base.Prolongation
                else:
                    operator_type = base.Operator
                op_info = initialization.OperatorInfo(name, level, constant.Stencil(stencil_entries, dimension),
                                                      operator_type)
                operators.append(op_info)
            elif tokens[0] == 'Equation':
                equation_name, level = tokens[1].split('@')
                eq_info = initialization.EquationInfo(equation_name, int(level), file.readline())
                if solution_equations is None or equation_name in solution_equations:
                    equations.append(eq_info)
                file.readline()
            line = file.readline()
    assert len(equations) > 0 and len(operators) > 0, "No equations specified"
    # max_level = max(equations, key=lambda info: info.level).level
    # equations = [eq_info for eq_info in equations if eq_info.level == max_level]
    equations.sort(key=lambda info: info.name)
    names_of_used_operators = set()
    for eq_info in equations:
        sympy_expr = eq_info.sympy_expr
        for symbol in sympy_expr.free_symbols:
            if any(symbol.name == op_info.name and eq_info.level == op_info.level for op_info in operators):
                names_of_used_operators.add(symbol.name + '@' + str(eq_info.level))
            elif symbol not in fields:
                fields.append(symbol)
    fields.sort(key=lambda s: s.name)
    for eq_info in equations:
        rhs_name = eq_info.rhs_name
        tmp = rhs_name.split('_')
        if len(tmp) == 1:
            if len(fields) != 1:
                raise RuntimeError('Could not extract associated field for rhs')
            eq_info._associated_field = fields[0]
        else:
            field_name = tmp[1]
            eq_info._associated_field = next(field for field in fields if field.name == field_name)
    equations.sort(key=lambda ei: ei.associated_field.name)
    for op_info in operators:
        if op_info.operator_type == base.Restriction or op_info.operator_type == base.Prolongation:
            name = op_info.name
            tmp = name.split('_')
            field_name = tmp[-1]
            try:
                associated_field = next(field for field in fields if field.name == field_name)
                op_info._associated_field = associated_field
                names_of_used_operators.add(op_info.name + '@' + str(op_info.level))
            except StopIteration as _:
                pass
    operators = [op_info for op_info in operators if op_info.name + '@' + str(op_info.level) in names_of_used_operators]

    # assert len(equations) == len(fields), 'The number of equations does not match with the number of fields'
    return equations, operators, fields


def extract_knowledge_information(base_path: str, relative_file_path: str):
    with open(f'{base_path}/{relative_file_path}', 'r') as file:
        for line in file:
            tokens = line.split('=')
            lhs = tokens[0].strip(' \n\t')
            if lhs == 'dimensionality':
                dimension = int(tokens[1].strip(' \n\t'))
            elif lhs == 'minLevel':
                min_level = int(tokens[1].strip(' \n\t'))
            elif lhs == 'maxLevel':
                max_level = int(tokens[1].strip(' \n\t'))
    return dimension, min_level, max_level


def extract_settings_information(base_path: str, relative_file_path: str):
    with open(f'{base_path}/{relative_file_path}', 'r') as file:
        for line in file:
            tokens = line.split('=')
            lhs = tokens[0].strip(' \n\t')
            if lhs == 'configName':
                config_name = tokens[1].strip(' \n\t"')
            elif lhs == 'basePathPrefix':
                base_path_prefix = tokens[1].strip(' \n\t"')
            elif lhs == 'debugL3File':
                debug_l3_path = tokens[1].strip(' \n\t"')
            elif lhs == 'outputPath':
                output_path = tokens[1].strip(' \n\t"')
        debug_l3_path = f"{base_path_prefix}/{debug_l3_path.replace('$configName$', config_name)}"
        output_path = f"{base_path_prefix}/{output_path.replace('$configName$', config_name)}"
    return base_path_prefix, config_name, debug_l3_path, output_path


def extract_nonlinear_term(base_path: str, relative_file_path: str, stencil_nonlinear: str, solution_field: str):
    def simplify_expr(ilayer4: str):  # remove field modifiers
        ilayer4 = re.sub('<[^>]+>', '', ilayer4)  # remove slot modifiers
        ilayer4 = re.sub('@[^)]+', '', ilayer4)  # remove modifiers starting with @
        return ilayer4

    # get layer4 expression of the nonlinear term
    stencil_expr = ""
    with open(f'{base_path}/{relative_file_path}', 'r') as file:
        line = file.readline()
        while line:
            tokens = line.split(' ')
            if tokens[0] == "Stencil" and stencil_nonlinear in tokens[1]:
                line = file.readline()
                tokens = line.split("=>")
                stencil_expr = tokens[1]
                line = False
            else:
                line = file.readline()

    nonlinear_expr = simplify_expr(stencil_expr.strip('\n') + " * " + solution_field)

    # convert to sympy
    return parse_expr(nonlinear_expr)
