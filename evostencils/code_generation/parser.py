from sympy.parsing.sympy_parser import parse_expr
from evostencils.expressions import base,  multigrid
import re
from evostencils.stencils import constant


class OperatorInfo:
    def __init__(self, name, level, stencil):
        self._name = name
        self._level = level
        self._stencil = stencil
        self._associated_field = None
        if 'gen_restrictionForSol_' in name:
            self._operator_type = multigrid.Restriction
        elif 'gen_prolongationForSol_' in name:
            self._operator_type = multigrid.Prolongation
        else:
            self._operator_type = base.Operator

    @property
    def name(self):
        return self._name

    @property
    def level(self):
        return self._level

    @property
    def stencil(self):
        return self._stencil

    @property
    def operator_type(self):
        return self._operator_type


class EquationInfo:
    def __init__(self, name: str, level: int, expr_str: str):
        self._name = name
        self._level = level
        transformed_expr = ''
        tokens = expr_str.split(' ')
        for token in tokens:
            transformed_expr += ' ' + token.split('@')[0]
        tmp = transformed_expr.split('==')
        self._sympy_expr = parse_expr(tmp[0])
        self._rhs_name = tmp[1]
        self._associated_field = None

    @property
    def name(self):
        return self._name

    @property
    def level(self):
        return self._level

    @property
    def sympy_expr(self):
        return self._sympy_expr

    @property
    def rhs_name(self):
        return self._rhs_name

    @property
    def associated_field(self):
        return self._associated_field


def parse_stencil_offsets(string):
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


def extract_l2_information(file_path, dimensionality):
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
                op_info = OperatorInfo(name, level, constant.Stencil(stencil_entries, dimensionality))
                operators.append(op_info)
            elif tokens[0] == 'Equation':
                tmp = tokens[1].split('@')
                eq_info = EquationInfo(tmp[0], int(tmp[1]), file.readline())
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
        field_name = tmp[1]
        eq_info._associated_field = next(field for field in fields if field.name == field_name)
    equations.sort(key=lambda ei: ei.associated_field.name)
    for op_info in operators:
        if op_info.operator_type == multigrid.Restriction or op_info.operator_type == multigrid.Prolongation:
            name = op_info.name
            tmp = name.split('_')
            field_name = tmp[-1]
            op_info._associated_field = next(field for field in fields if field.name == field_name)
    assert len(equations) == len(fields), 'The number of equations does not match with the number of fields'
    return equations, operators, fields


def extract_knowledge_information(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.split('=')
            lhs = tokens[0].strip(' \n\t')
            if lhs == 'dimensionality':
                dimensionality = int(tokens[1].strip(' \n\t'))
            elif lhs == 'minLevel':
                min_level = int(tokens[1].strip(' \n\t'))
            elif lhs == 'maxLevel':
                max_level = int(tokens[1].strip(' \n\t'))
    return dimensionality, min_level, max_level


def extract_settings_information(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.split('=')
            lhs = tokens[0].strip(' \n\t')
            if lhs == 'configName':
                config_name = tokens[1].strip(' \n\t"')
            elif lhs == 'basePathPrefix':
                base_path = tokens[1].strip(' \n\t"')
    return base_path, config_name

#extract_layer_2_information('/home/jonas/Schreibtisch/exastencils/Examples/Debug/3D_FV_Stokes_fromL2_debug.exa2', 3)
extract_l2_information('/home/jonas/Schreibtisch/exastencils/Examples/Debug/2D_FD_Stokes_fromL2_debug.exa2', 2)
#extract_layer_2_information('/home/jonas/Schreibtisch/exastencils/Examples/Debug/2D_FD_OptFlow_fromL2_debug.exa2', 2)
#extract_layer_2_information('/home/jonas/Schreibtisch/exastencils/Examples/Debug/2D_FD_Poisson_fromL2_debug.exa3', 2)
extract_knowledge_information('/home/jonas/Schreibtisch/exastencils/Examples/Stokes/2D_FD_Stokes_fromL2.knowledge')
extract_settings_information('/home/jonas/Schreibtisch/exastencils/Examples/Stokes/2D_FD_Stokes_fromL2.settings')

