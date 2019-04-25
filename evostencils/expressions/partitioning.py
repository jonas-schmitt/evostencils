import abc
import evostencils.stencils.constant as constant
import evostencils.stencils.periodic as periodic
from evostencils.expressions import base, multigrid as mg


class Partitioning(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def generate(_):
        pass


class Single:
    @staticmethod
    def generate(stencil, grid):
        if stencil is None:
            return [None]
        else:
            return [constant.get_unit_stencil(grid)]

    def __repr__(self):
        return 'Single()'


class RedBlack:
    @staticmethod
    def generate(stencil, grid):
        if stencil is None:
            return [None]
        else:
            return periodic.red_black_partitioning(stencil, grid)

    def __repr__(self):
        return 'RedBlack()'


def can_be_partitioned(expression: base.Expression):
    if mg.is_intergrid_operation(expression):
        return False
    elif isinstance(expression, base.BinaryExpression):
        return can_be_partitioned(expression.operand1) and can_be_partitioned(expression.operand2)
    elif isinstance(expression, base.UnaryScalarExpression) or isinstance(expression, base.Scaling):
        return can_be_partitioned(expression.operand)
    elif isinstance(expression, base.Operator):
        return True
    else:
        return False
