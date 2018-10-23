from evostencils.expressions import base, multigrid as mg


class Field:
    def __init__(self, expression, name=None):
        self.expression = expression
        self.name = name

def obtain_coarsest_grid_size(expression: base.Expression) -> tuple:

    if isinstance(expression, mg.Cycle):


# Warning: This function modifies the expression passed to it
def identify_temporary_fields(node: base.Expression) -> list:
    declarations = []
    if isinstance(node, mg.Cycle):
        declarations.extend(identify_temporary_fields(node.iterate))
        declarations.extend(identify_temporary_fields(node.correction))
        # Reuse the solution field here
        declarations.append(Field(node))
        node.storage = declarations[-1]
    elif isinstance(node, base.BinaryExpression):
        if isinstance(node.operand1, base.Grid) or isinstance(node.operand1, mg.Cycle):
            declarations.append(Field(node))
            node.storage = declarations[-1]
        elif mg.contains_intergrid_operation(node.operand1):
            declarations.append(Field(node))
            node.storage = declarations[-1]
            declarations.extend(identify_temporary_fields(node.operand1))
        elif isinstance(node.operand2, base.Grid) or isinstance(node.operand1, mg.Cycle):
            declarations.append(Field(node))
            node.storage = declarations[-1]
        declarations.extend(identify_temporary_fields(node.operand1))
        declarations.extend(identify_temporary_fields(node.operand2))
    elif isinstance(node, base.UnaryExpression) or isinstance(node, base.Scaling):
        declarations.extend(identify_temporary_fields(node.operand))
    return declarations


def name_fields(field_declarations):
    for i, field in enumerate(field_declarations):
        field.name = f'tmp_{i}'


def print_declarations(field_declarations):
    for field in field_declarations:
        print(f'Field {field.name}@ with Real on Node of global = 0.0\n')
