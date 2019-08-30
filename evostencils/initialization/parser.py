from sympy.parsing.sympy_parser import parse_expr

from evostencils.expressions import base
import re
from evostencils.stencils import constant
from evostencils.initialization import multigrid as initialization


def parse_stencil_offsets(string: str):
    tokens = re.split(r" ?\[ ?| ?\] ?| ?, ?", string)
    tokens = [token for token in tokens if not token == ""]
    offsets = []
    for token in tokens:
        sympy_expr = parse_expr(token)
        substituted_expr = sympy_expr
        for symbol in sympy_expr.free_symbols:
            substituted_expr = substituted_expr.subs(symbol, 0)
        offset = int(round(substituted_expr.evalf()))
        offsets.append(offset)
    return tuple(offsets)


def extract_l2_information(file_path: str, dimension: int):
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
                while True:
                    tmp = tokens[1].split('with')
                    value = float(tmp[1])
                    offsets = parse_stencil_offsets(tmp[0])
                    stencil_entries.append((offsets, value))
                    line = file.readline()
                    tokens = line.split('from')
                    if '}' in line:
                        break
                if 'gen_restrictionForSol_' in name:
                    operator_type = base.Restriction
                elif 'gen_prolongationForSol_' in name:
                    operator_type = base.Prolongation
                else:
                    operator_type = base.Operator
                op_info = initialization.OperatorInfo(name, level, constant.Stencil(stencil_entries, dimension),
                                                 operator_type)
                operators.append(op_info)
            elif tokens[0] == 'Equation':
                tmp = tokens[1].split('@')
                eq_info = initialization.EquationInfo(tmp[0], int(tmp[1]), file.readline())
                equations.append(eq_info)
                file.readline()
            line = file.readline()
    max_level = max(equations, key=lambda info: info.level).level
    equations = [eq_info for eq_info in equations if eq_info.level == max_level]
    equations.sort(key=lambda info: info.name)
    for eq_info in equations:
        sympy_expr = eq_info.sympy_expr
        for symbol in sympy_expr.free_symbols:
            if symbol.name not in (op_info.name for op_info in operators) and symbol not in fields:
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
            op_info._associated_field = next(field for field in fields if field.name == field_name)
    assert len(equations) == len(fields), 'The number of equations does not match with the number of fields'
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


def generate_level_adapted_knowledge_file(base_path: str, relative_input_file_path: str, relative_output_file_path: str,
                                          min_level: int, max_level: int, l3_file_name: str):
    with open(f'{base_path}/{relative_input_file_path}', 'r') as input_file:
        with open(f'{base_path}{relative_output_file_path}', 'w') as output_file:
            for line in input_file:
                tokens = line.split('=')
                lhs = tokens[0].strip(' \n\t')
                if lhs == 'minLevel':
                    output_file.write(f'{lhs}\t= {min_level}\n')
                elif lhs == 'maxLevel':
                    output_file.write(f'{lhs}\t= {max_level}\n')
                elif lhs == 'l3file':
                    pass
                else:
                    output_file.write(line)
            output_file.write(f'l3file\t= {l3_file_name}')


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
