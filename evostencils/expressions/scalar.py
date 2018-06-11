from sympy import MatrixSymbol
import sympy as sp
from functools import reduce
import operator


class SparseMatrixSymbol(MatrixSymbol):
    def __new__(cls, name, m, n):
        m, n = sp.sympify(m), sp.sympify(n)
        obj = super().__new__(cls, name, m, n)
        obj._source_matrix = None
        return obj

    def set_source_matrix(self, source_matrix):
        self._source_matrix = source_matrix

    @property
    def get_source_matrix(self):
        return self._source_matrix


class DiagonalMatrixSymbol(SparseMatrixSymbol):
    pass


class LowerTriangularMatrixSymbol(SparseMatrixSymbol):
    pass


class UpperTriangularMatrixSymbol(SparseMatrixSymbol):
    pass


def get_diagonal(A) -> SparseMatrixSymbol:
    D = DiagonalMatrixSymbol(f"{A.name}_d", *A.shape)
    D.set_source_matrix(A)
    return D


def get_lower_triangle(A) -> SparseMatrixSymbol:
    L = LowerTriangularMatrixSymbol(f"{A.name}_l", *A.shape)
    L.set_source_matrix(A)
    return L


def get_upper_triangle(A) -> SparseMatrixSymbol:
    U = UpperTriangularMatrixSymbol(f"{A.name}_u", *A.shape)
    U.set_source_matrix(A)
    return U


def generate_vector_on_grid(name: str, grid_size: tuple) -> MatrixSymbol:
    n = reduce(operator.mul, grid_size, 1)
    return MatrixSymbol(name, n, 1)


def generate_matrix_on_grid(name: str, grid_size: tuple) -> MatrixSymbol:
    n = reduce(operator.mul, grid_size, 1)
    return MatrixSymbol(name, n, n)
