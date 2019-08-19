from sympy.parsing.sympy_parser import parse_expr

class OperatorInfo:
    def __init__(self, name, level, stencil):
        self._name = name
        self._level = level
        self._stencil = stencil

    @property
    def name(self):
        return self._name

    @property
    def level(self):
        return self._level

    @property
    def stencil(self):
        return self._stencil

class EquationInfo:
    def __init__(self, name: str, level: int, expr_str: str):
        self._name = name
        self._level = level
        transformed_expr = ''
        tokens = expr_str.split(' ')
        for token in tokens:
            transformed_expr += ' ' + token.split('@')[0]
        lhs = transformed_expr.split('==')[0]
        self._sympy_expr = parse_expr(lhs)

    @property
    def name(self):
        return self._name

    @property
    def level(self):
        return self._level

    @property
    def sympy_expr(self):
        return self._sympy_expr


def extract_layer_2_information(file_path):
    equations = []
    operators = []
    fields = []
    with open(file_path, 'r') as file:
        line = file.readline()
        while line:
            tokens = line.split(' ')
            if tokens[0] == 'Operator':
                tmp = tokens[1].split('@')
                op_info = OperatorInfo(tmp[0], int(tmp[1]), None)
                #TODO parse stencil
                operators.append(op_info)
            elif tokens[0] == 'Equation':
                tmp = tokens[1].split('@')
                eq_info = EquationInfo(tmp[0], int(tmp[1]), file.readline())
                equations.append(eq_info)
                file.readline()
            line = file.readline()
    max_level = max(equations, key=lambda info: info._level).level
    equations = [eq_info for eq_info in equations if eq_info.level == max_level]
    equations.sort(key=lambda info: info.name)
    for eq_info in equations:
        sympy_expr = eq_info.sympy_expr
        for symbol in sympy_expr.free_symbols:
            if symbol.name not in (op_info.name for op_info in operators) and symbol not in fields:
                fields.append(symbol)
    return equations, operators, fields

#extract_layer_2_information('/home/jonas/Schreibtisch/exastencils/Examples/Debug/3D_FV_Stokes_fromL2_debug.exa2')
#extract_layer_2_information('/home/jonas/Schreibtisch/exastencils/Examples/Debug/2D_FD_Stokes_fromL2_debug.exa2')
#extract_layer_2_information('/home/jonas/Schreibtisch/exastencils/Examples/Debug/2D_FD_OptFlow_fromL2_debug.exa2')
