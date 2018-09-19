import abc
from operator import mul as builtin_mul
from functools import reduce
from evostencils.stencils import periodic
from evostencils.stencils import constant


# Base classes
class Expression(abc.ABC):
    @property
    @abc.abstractmethod
    def shape(self):
        pass

    @abc.abstractmethod
    def apply(self, transform: callable):
        pass


class Entity(Expression):
    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    def __str__(self):
        return f'{self.name}'

    def apply(self, _, *args):
        return self


class UnaryExpression(Expression):
    def __init__(self, operand):
        self._operand = operand
        self._shape = operand.shape

    @property
    def operand(self):
        return self._operand

    @property
    def shape(self):
        return self._shape

    def apply(self, transform: callable, *args):
        return type(self)(transform(self.operand, *args))


class BinaryExpression(Expression):
    @property
    def operand1(self):
        return self._operand1

    @property
    def operand2(self):
        return self._operand2

    @property
    def shape(self):
        return self._shape

    def apply(self, transform: callable, *args):
        return type(self)(transform(self.operand1, *args), transform(self.operand2, *args))


# Entities
class Operator(Entity):
    def __init__(self, name, shape, stencil=None):
        self._name = name
        self._shape = shape
        self._stencil = stencil

    def generate_stencil(self):
        return self._stencil

    def __repr__(self):
        return f'Operator({repr(self.name)}, {repr(self.shape)}, {repr(self.generate_stencil())})'


class Identity(Operator):
    def __init__(self, shape, dimension):
        self._dimension = dimension
        super(Identity, self).__init__('I', shape, constant.get_unit_stencil(dimension))

    @property
    def dimension(self):
        return self._dimension

    def __repr__(self):
        return f'Identity({repr(self.shape)}, {repr(self.dimension)})'


class Zero(Operator):
    def __init__(self, shape):
        super(Zero, self).__init__('0', shape, constant.get_null_stencil())

    def __repr__(self):
        return f'Zero({repr(self.shape)})'


class Grid(Entity):
    def __init__(self, name, size):
        import operator
        self._name = name
        self._size = size
        self._shape = (reduce(operator.mul, size), 1)

    @property
    def size(self):
        return self._size

    @property
    def dimension(self):
        return len(self.size)

    @staticmethod
    def generate_stencil():
        return None

    def __repr__(self):
        return f'Grid({repr(self.name)}, {repr(self.size)})'


# Unary Expressions
class Diagonal(UnaryExpression):
    def generate_stencil(self):
        return periodic.diagonal(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.diag'

    def __repr__(self):
        return f'Diagonal({repr(self.operand)})'


class LowerTriangle(UnaryExpression):
    def generate_stencil(self):
        return periodic.lower(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.lower'

    def __repr__(self):
        return f'LowerTriangle({repr(self.operand)})'


class UpperTriangle(UnaryExpression):
    def generate_stencil(self):
        return periodic.upper(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.upper'

    def __repr__(self):
        return f'UpperTriangle({repr(self.operand)})'


class BlockDiagonal(UnaryExpression):
    def __init__(self, operand, block_size):
        self._block_size = block_size
        super(BlockDiagonal, self).__init__(operand)

    def generate_stencil(self):
        return periodic.block_diagonal(self.operand.generate_stencil(), self.block_size)

    @property
    def block_size(self):
        return self._block_size

    def __str__(self):
        return f'{str(self.operand)}.block_diag'

    def __repr__(self):
        return f'BlockDiagonal({repr(self.operand)}, {repr(self.block_size)})'

    def apply(self, transform: callable, *args):
        return type(self)(transform(self.operand, *args), self.block_size)


class Inverse(UnaryExpression):
    def generate_stencil(self):
        return periodic.inverse(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.I'

    def __repr__(self):
        return f'Inverse({repr(self.operand)})'


class Transpose(UnaryExpression):
    def __init__(self, operand):
        self._operand = operand
        self._shape = (operand.shape[1], operand.shape[0])

    def generate_stencil(self):
        return periodic.transpose(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.T'

    def __repr__(self):
        return f'Transpose({repr(self.operand)})'


# Binary Expressions
class Addition(BinaryExpression):
    def __init__(self, operand1, operand2):
        assert operand1.shape == operand2.shape, "Operand shapes are not equal"
        self._operand1 = operand1
        self._operand2 = operand2
        self._shape = operand1.shape

    def generate_stencil(self):
        return periodic.add(self.operand1.generate_stencil(), self.operand2.generate_stencil())

    def __str__(self):
        return f'({str(self.operand1)} + {str(self.operand2)})'

    def __repr__(self):
        return f'Addition({repr(self.operand1)}, {repr(self.operand2)})'


class Subtraction(BinaryExpression):
    def __init__(self, operand1, operand2):
        assert operand1.shape == operand2.shape, "Operand shapes are not equal"
        self._operand1 = operand1
        self._operand2 = operand2
        self._shape = operand1.shape

    def generate_stencil(self):
        return periodic.sub(self.operand1.generate_stencil(), self.operand2.generate_stencil())

    def __str__(self):
        return f'({str(self.operand1)} - {str(self.operand2)})'

    def __repr__(self):
        return f'Subtraction({repr(self.operand1)}, {repr(self.operand2)})'


class Multiplication(BinaryExpression):
    def __init__(self, operand1, operand2):
        assert operand1.shape[1] == operand2.shape[0], "Operand shapes are not aligned"
        self._operand1 = operand1
        self._operand2 = operand2
        self._shape = (operand1.shape[0], operand2.shape[1])

    def generate_stencil(self):
        return periodic.mul(self.operand1.generate_stencil(), self.operand2.generate_stencil())

    def __str__(self):
        return f'({str(self.operand1)} * {str(self.operand2)})'

    def __repr__(self):
        return f'Multiplication({repr(self.operand1)}, {repr(self.operand2)})'


# Scaling
class Scaling(Expression):
    def __init__(self, factor, operand):
        self._factor = factor
        self._operand = operand
        self._shape = operand.shape

    @property
    def factor(self):
        return self._factor

    @property
    def operand(self):
        return self._operand

    @property
    def shape(self):
        return self._shape

    def generate_stencil(self):
        return periodic.scale(self.factor, self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.factor)} * {str(self.operand)}'

    def __repr__(self):
        return f'Scaling({repr(self.factor)}, {repr(self.operand)})'

    def apply(self, transform: callable, *args):
        return Scaling(self.factor, transform(self.operand, *args))


class Partitioning(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def generate(_):
        pass


class NonePartitioning:
    @staticmethod
    def generate(stencil):
        if stencil is None:
            return [None]
        else:
            return [constant.get_unit_stencil(stencil.dimension)]


class RedBlackPartitioning:
    @staticmethod
    def generate(stencil):
        if stencil is None:
            return [None]
        else:
            return periodic.red_black_partitioning(stencil)


# Wrapper functions
def inv(operand):
    return Inverse(operand)


def add(operand1, operand2):
    return Addition(operand1, operand2)


def sub(operand1, operand2):
    return Addition(operand1, operand2)


def mul(operand1, operand2):
    return Multiplication(operand1, operand2)


def scale(factor, operand):
    return Scaling(factor, operand)


def generate_grid(name: str, grid_size: tuple) -> Grid:
    return Grid(name, grid_size)


def generate_operator(name: str, grid_size: tuple, stencil=None) -> Operator:
    n = reduce(builtin_mul, grid_size, 1)
    return Operator(name, (n, n), stencil)


def is_quadratic(expression: Expression) -> bool:
    return expression.shape[0] == expression.shape[1]

