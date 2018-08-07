import abc
from operator import mul as builtin_mul
from functools import reduce
from evostencils import stencils as stencils


# Base classes
class Expression(abc.ABC):
    @property
    @abc.abstractmethod
    def shape(self):
        pass


class Entity(Expression):
    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    def __repr__(self):
        return f'{self.__class__}({self.name}, {self.shape})'

    def __str__(self):
        return f'{self.name}'


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

    def __repr__(self):
        return f'{self.__class__}({self.operand})'


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

    def __repr__(self):
        return f'{self.__class__}({self.operand1}, {self.operand2})'


# Entities
class Operator(Entity):
    def __init__(self, name, shape, stencil=None):
        self._name = name
        self._shape = shape
        self._stencil = stencil

    def generate_stencil(self):
        return self._stencil


class Identity(Operator):
    def __init__(self, shape, dimension=None):
        super(Identity, self).__init__('I', shape, stencils.get_unit_stencil(dimension))


class Zero(Operator):
    def __init__(self, shape):
        super(Zero, self).__init__('0', shape, stencils.get_null_stencil())


class Grid(Entity):
    def __init__(self, name, size):
        self._name = name
        self._shape = (size, 1)

    @property
    def size(self):
        return self._shape[0]

    @staticmethod
    def generate_stencil():
        return None


# Unary Expressions
class Diagonal(UnaryExpression):
    def generate_stencil(self):
        return stencils.diagonal(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.diag'


class LowerTriangle(UnaryExpression):
    def generate_stencil(self):
        return stencils.lower(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.lower'


class UpperTriangle(UnaryExpression):
    def generate_stencil(self):
        return stencils.upper(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.upper'


class Inverse(UnaryExpression):
    def generate_stencil(self):
        return stencils.inverse(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.I'


class Transpose(UnaryExpression):
    def __init__(self, operand):
        self._operand = operand
        self._shape = (operand.shape[1], operand.shape[0])

    def generate_stencil(self):
        return stencils.transpose(self.operand.generate_stencil())

    def __str__(self):
        return f'{str(self.operand)}.T'


# Binary Expressions
class Addition(BinaryExpression):
    def __init__(self, operand1, operand2):
        assert operand1.shape == operand2.shape, "Operand shapes are not equal"
        self._operand1 = operand1
        self._operand2 = operand2
        self._shape = operand1.shape

    def generate_stencil(self):
        return stencils.add(self.operand1.generate_stencil(), self.operand2.generate_stencil())

    def __str__(self):
        return f'({str(self.operand1)} + {str(self.operand2)})'


class Subtraction(BinaryExpression):
    def __init__(self, operand1, operand2):
        assert operand1.shape == operand2.shape, "Operand shapes are not equal"
        self._operand1 = operand1
        self._operand2 = operand2
        self._shape = operand1.shape

    def generate_stencil(self):
        return stencils.sub(self.operand1.generate_stencil(), self.operand2.generate_stencil())

    def __str__(self):
        return f'({str(self.operand1)} - {str(self.operand2)})'


class Multiplication(BinaryExpression):
    def __init__(self, operand1, operand2):
        assert operand1.shape[1] == operand2.shape[0], "Operand shapes are not aligned"
        self._operand1 = operand1
        self._operand2 = operand2
        self._shape = (operand1.shape[0], operand2.shape[1])

    def generate_stencil(self):
        return stencils.mul(self.operand1.generate_stencil(), self.operand2.generate_stencil())

    def __str__(self):
        return f'({str(self.operand1)} * {str(self.operand2)})'


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
        return stencils.scale(self.factor, self.operand.generate_stencil())

    def __repr__(self):
        return f'{self.__class__}({self.factor}, {self.operand})'

    def __str__(self):
        return f'{str(self.factor)} * {str(self.operand)}'


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
    n = reduce(builtin_mul, grid_size, 1)
    return Grid(name, n)


def generate_operator(name: str, grid_size: tuple, stencil=None) -> Operator:
    n = reduce(builtin_mul, grid_size, 1)
    return Operator(name, (n, n), stencil)
