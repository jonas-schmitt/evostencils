import sympy as sp
from sympy import MatrixSymbol
from functools import reduce
import operator


def diagonal(matrix_symbol: MatrixSymbol) -> MatrixSymbol:
    return MatrixSymbol(str(matrix_symbol) + ".diag", *matrix_symbol.shape)


def lower(matrix_symbol: MatrixSymbol) -> MatrixSymbol:
    return MatrixSymbol(str(matrix_symbol) + ".lower", *matrix_symbol.shape)


def upper(matrix_symbol: MatrixSymbol) -> MatrixSymbol:
    return MatrixSymbol(str(matrix_symbol) + ".upper", *matrix_symbol.shape)


def generate_vector_on_grid(name: str, grid_size: tuple) -> MatrixSymbol:
    n = reduce(operator.mul, grid_size, 1)
    return MatrixSymbol(name, n, 1)


def generate_matrix_on_grid(name: str, grid_size: tuple) -> MatrixSymbol:
    n = reduce(operator.mul, grid_size, 1)
    return MatrixSymbol(name, n, n)


class MatrixMetaClass(type):
    def __new__(mcs, class_name, bases, dct):
        return super(MatrixMetaClass, mcs).__new__(mcs, class_name, bases, dct)

    def __eq__(self, other):
        return self._invertible == other._invertible and self._shape == other._shape

    def __hash__(self):
        return hash((*self._shape, self._invertible))


def generate_operator_type(shape, invertible):
    return MatrixMetaClass("MatrixType", (), {"_shape": shape, "_invertible": invertible})
