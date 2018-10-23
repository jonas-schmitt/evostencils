from evostencils.expressions import base, multigrid as mg


class Field:
    def __init__(self, expression):
        self._expression = expression

    @property
    def expression(self):
        return self._expression


def identify_temporary_fields(node: base.Expression) -> list:
    declarations = []
    if isinstance(node, mg.Cycle):
        declarations.extend(identify_temporary_fields(node.iterate))
        declarations.extend(identify_temporary_fields(node.correction))
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
