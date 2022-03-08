# --------------------------------STATEMENTS AND EXPRESSIONS------------------------------------
class Addition:
    def __init__(self, summands):
        self.summands = summands

    def pretty_print(self):
        return "( " + " + ".join(print_exa(s) for s in self.summands) + " )"


class Subtraction:
    def __init__(self, minuend, subtrahend):
        self.minuend = minuend
        self.subtrahend = subtrahend

    def pretty_print(self):
        return f'( {print_exa(self.minuend)} - {print_exa(self.subtrahend)} )'


class Multiplication:
    def __init__(self, factors):
        self.factors = factors

    def pretty_print(self):
        return "( " + " * ".join(print_exa(s) for s in self.factors) + " )"


class Division:
    def __init__(self, dividend, divisor):
        self.dividend = dividend
        self.divisor = divisor

    def pretty_print(self):
        return f'( {print_exa(self.dividend)} / {print_exa(self.divisor)} )'


class FunctionCall:
    def __init__(self, fct_name: str, args=None):
        if args is None:
            args = []
        self.fct_name = fct_name
        self.args = args

    def pretty_print(self):
        return self.fct_name + " ( " + (", ".join(print_exa(s) for s in self.args)) + " )"


class VariableDecl:
    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type

    def pretty_print(self):
        return f'Variable {self.name} : {self.type}'


class Assignment:
    def __init__(self, lhs, rhs, op="="):
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def pretty_print(self):
        return f'{print_exa(self.lhs)} {self.op} {print_exa(self.rhs)}'


class Update:
    def __init__(self, lhs, rhs, op="+="):
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def pretty_print(self):
        return f'{print_exa(self.lhs)} {self.op} {print_exa(self.rhs)}'


class Communicate:
    def __init__(self, field):
        self.field = field

    def pretty_print(self):
        return f'communicate {print_exa(self.field)}'


class ApplyBC:
    def __init__(self, field):
        self.field = field

    def pretty_print(self):
        return f'apply bc to {print_exa(self.field)}'


class Advance:
    def __init__(self, field):
        self.field = field

    def pretty_print(self):
        return f'advance {print_exa(self.field)}'


class FieldLoop:
    def __init__(self, field, body):
        self.field = field
        self.body = body

    def pretty_print(self):
        expr = f'loop over {print_exa(self.field)}' \
               + ' {\n' \
               + '  ' + ('\n  '.join(print_exa(s) for s in self.body)) + \
               '\n}'
        return expr


class ApplyColor:
    def __init__(self, colors, body):
        self.colors = colors
        self.body = body

    def pretty_print(self):
        expr = f'color with' + ' {\n' \
               + '  ' + (',\n  '.join(print_exa(s) for s in self.colors)) + ','\
               + '\n  ' + ('\n  '.join(print_exa(s) for s in self.body))  \
               + '\n}'
        return expr


class Smoothing:
    def __init__(self, field, correction, relaxation_factor):
        self.field = field
        self.correction = correction
        self.relaxation_factor = relaxation_factor

    def pretty_print(self):
        return f'{print_exa(self.field)} += {str(self.relaxation_factor)} * {print_exa(self.correction)}'


# -------------------FIELDS AND STENCILS---------------------------------
class Field:
    def __init__(self, field_name, level):
        self.field_name = field_name
        self.lvl = level

    def pretty_print(self):
        return f'{self.field_name}@{self.lvl}'


class FieldSlotted:
    def __init__(self, field, slot):
        self.field = field
        self.slot = slot

    def pretty_print(self):
        return f'{self.field.field_name}<{self.slot}>@{self.field.lvl}'


class Stencil:
    def __init__(self, stencil_name, level):
        self.stencil_name = stencil_name
        self.lvl = level

    def pretty_print(self):
        return f'{self.stencil_name}@{self.lvl}'


class StencilElement:
    def __init__(self, stencil, modifier):
        self.stencil = stencil
        self.modifier = modifier

    def pretty_print(self):
        return f'{self.stencil.stencil_name}@{self.stencil.lvl}:{self.modifier}'


# --------------------------------FUNCTIONS---------------------------------------
class Function:
    def __init__(self, name: str, body, parameters=None, levels=""):
        self.name = name
        self.body = body
        if parameters is None:
            parameters = []
        self.parameters = parameters
        self.levels = levels

    def pretty_print(self):
        fct_print = "Function " + self.name + self.levels + " ( " + ", ".join(print_exa(p) for p in self.parameters) + " )"
        fct_print += " {\n" + \
                     "" + ("\n".join(print_exa(s) for s in self.body)) + \
                     "\n}\n\n"

        return fct_print


# ---------------------------------PRINTER----------------------------------

def print_exa(expr):
    if type(expr).__name__ == 'str':
        return expr
    elif 'float' in type(expr).__name__:
        return str(expr)
    else:
        return expr.pretty_print()
