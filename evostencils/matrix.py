#import sympy as sp
from sympy import MatrixSymbol
from functools import reduce
import operator


class VectorType(MatrixSymbol):
    def __new__(cls, name, m, n):
        obj = super().__new__(cls, name, m, 1)
        return obj


class MatrixType(MatrixSymbol):
    def __new__(cls, name, m, n):
        obj = super().__new__(cls, name, m, n)
        return obj

class SplittedMatrix(MatrixSymbol):
    def __new__(cls, name, source_matrix):
        obj = super().__new__(cls, name, source_matrix.shape[0], source_matrix.shape[1])
        obj._source_matrix = source_matrix
        return obj

    @property
    def get_source_matrix(self):
        return self._source_matrix


class DiagonalMatrixType(SplittedMatrix):
    def __new__(cls, source_matrix):
        return super().__new__(cls, f"{source_matrix.name}.diag", source_matrix)


class TriangularMatrixType(SplittedMatrix):
    def __new__(cls, name, source_matrix):
        obj = super().__new__(cls, name, source_matrix)
        return obj


class LowerTriangularMatrixType(TriangularMatrixType):
    def __new__(cls, source_matrix):
        return super().__new__(cls, f"{source_matrix.name}.lower", source_matrix)


class UpperTriangularMatrixType(TriangularMatrixType):
    def __new__(cls, source_matrix):
        return super().__new__(cls, f"{source_matrix.name}.upper", source_matrix)


def generate_vector_on_grid(name: str, grid_size: tuple) -> VectorType:
    n = reduce(operator.mul, grid_size, 1)
    return VectorType(name, n, 1)


def generate_matrix_on_grid(name: str, grid_size: tuple) -> MatrixType:
    n = reduce(operator.mul, grid_size, 1)
    return MatrixType(name, n, n)



# def diagonal(matrix_symbol: MatrixSymbol) -> MatrixSymbol:
#     return MatrixSymbol(str(matrix_symbol) + ".diag", *matrix_symbol.shape)
#
#
# def lower(matrix_symbol: MatrixSymbol) -> MatrixSymbol:
#     return MatrixSymbol(str(matrix_symbol) + ".lower", *matrix_symbol.shape)
#
#
# def upper(matrix_symbol: MatrixSymbol) -> MatrixSymbol:
#     return MatrixSymbol(str(matrix_symbol) + ".upper", *matrix_symbol.shape)
#
#
# class MatrixMetaClass(type):
#     def __new__(mcs, class_name, bases, dct):
#         return super(MatrixMetaClass, mcs).__new__(mcs, class_name, bases, dct)
#
#     def __eq__(self, other):
#         return self.invertible == other.invertible and self.shape == other.shape
#
#     def __hash__(self):
#         return hash((*self.shape, self.invertible))
#
#
# def generate_matrix_type(shape, invertible):
#     return MatrixMetaClass("MatrixType", (), {
#         "shape": shape,
#         "invertible": invertible
#     })
