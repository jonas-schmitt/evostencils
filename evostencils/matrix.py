from sympy import MatrixSymbol
from functools import reduce
import operator

#class SparseMatrixSymbol(MatrixSymbol):
#    def __new__(cls, name, m, n):
#        obj = super().__new__(cls, name, m, n)
#        obj._matrix_type = None
#        return obj
#
#    def set_matrix_type(self, matrix_type):
#        self._matrix_type = matrix_type
#
#    @property
#    def get_matrix_type(self):
#        return self._matrix_type


class SplittedMatrixSymbol(MatrixSymbol):
    def __new__(cls, name, m, n):
        obj = super().__new__(cls, name, m, n)
        obj._source_matrix = None
        return obj

    def set_source_matrix(self, source_matrix):
        self._source_matrix = source_matrix

    @property
    def get_source_matrix(self):
        return self._source_matrix


def get_diagonal(A) -> SplittedMatrixSymbol:
    D = SplittedMatrixSymbol(f"{A.name}.diagonal", *A.shape)
    D.set_source_matrix(A)
    return D


def get_lower_triangle(A) -> SplittedMatrixSymbol:
    L = SplittedMatrixSymbol(f"{A.name}.lower_triangle", *A.shape)
    L.set_source_matrix(A)
    return L


def get_upper_triangle(A) -> SplittedMatrixSymbol:
    U = SplittedMatrixSymbol(f"{A.name}.upper_triangle", *A.shape)
    U.set_source_matrix(A)
    return U


def generate_vector_on_grid(name: str, grid_size: tuple) -> MatrixSymbol:
    n = reduce(operator.mul, grid_size, 1)
    return MatrixSymbol(name, n, 1)


def generate_matrix_on_grid(name: str, grid_size: tuple) -> MatrixSymbol:
    n = reduce(operator.mul, grid_size, 1)
    return MatrixSymbol(name, n, n)


class MatrixTypeMetaClass(type):
    def __new__(mcs, class_name, bases, dct):
        return super(MatrixTypeMetaClass, mcs).__new__(mcs, class_name, bases, dct)

    def __eq__(self, other):
        return self.shape == other.shape \
               and self.diagonal == other.diagonal \
               and self.lower_triangle == other.lower_triangle \
               and self.upper_triangle == other.upper_triangle

    def __hash__(self):
        return hash((*self.shape, self.diagonal, self.lower_triangle, self.upper_triangle))


def generate_matrix_type(shape_, diag=True, lower_triangle=True, upper_triangle=True):
    return MatrixTypeMetaClass("MatrixType", (), {
        "shape": shape_,
        "diagonal": diag,
        "lower_triangle": lower_triangle,
        "upper_triangle": upper_triangle
    })
