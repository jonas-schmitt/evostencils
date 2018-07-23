from abc import ABC


class Expression(ABC):
    pass


class Operator(Expression):
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class Diagonal(Operator):
    def __init__(self, operator):
        super(Diagonal, self).__init__(operator.name, operator.shape)


class LowerTriangle(Operator):
    def __init__(self, operator):
        super(LowerTriangle, self).__init__(operator.name, operator.shape)


class UpperTriangle(Operator):
    def __init__(self, operator):
        super(UpperTriangle, self).__init__(operator.name, operator.shape)


class Identity(Operator):
    def __init__(self, grid):
        super(Identity, self).__init__('I', (grid.size, grid.size))


class Zero(Operator):
    def __init__(self, operator):
        super(Zero, self).__init__('0', operator.shape)


class Grid(Expression):
    def __init__(self, name, size):
        self.name = name
        self.shape = (size, 1)


class Addition(Expression):
    def __init__(self, operand1, operand2):
        self.operand1 = operand1
        self.operand2 = operand2


class Subtraction(Expression):
    def __init__(self, operand1, operand2):
        self.operand1 = operand1
        self.operand2 = operand2


class Multiplication(Expression):
    def __init__(self, operand1, operand2):
        self.operand1 = operand1
        self.operand2 = operand2


class Scaling(Expression):
    def __init__(self, factor, operand):
        self.factor = factor
        self.operand = operand


class Inverse(Expression):
    def __init__(self, operator):
        self.operator = operator


class Transposition(Expression):
    def __init__(self, operator):
        self.operator = operator


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

