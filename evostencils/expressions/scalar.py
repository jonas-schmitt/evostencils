from sympy import MatrixSymbol
from sympy.matrices.expressions.factorizations import Factorization
import sympy as sp
from functools import reduce
import operator


class Diagonal(Factorization):
    predicates = sp.Q.diagonal

    def __str__(self):
        return f"{self.arg}_d"


class Lower(Factorization):
    predicates = sp.Q.lower_triangular

    def __str__(self):
        return f"{self.arg}_l"


class Upper(Factorization):
    predicates = sp.Q.upper_triangular

    def __str__(self):
        return f"{self.arg}_u"


def split_matrix(expr):
    return Diagonal(expr), Lower(expr), Upper(expr)


def get_diagonal(expr) -> Diagonal:
    D, _, _ = split_matrix(expr)
    return D


def get_lower_triangle(expr) -> Lower:
    _, L, _ = split_matrix(expr)
    return L


def get_upper_triangle(expr) -> Upper:
    _, _, U = split_matrix(expr)
    return U


def generate_vector_on_grid(name: str, grid_size: tuple) -> MatrixSymbol:
    n = reduce(operator.mul, grid_size, 1)
    return MatrixSymbol(name, n, 1)


def generate_matrix_on_grid(name: str, grid_size: tuple) -> MatrixSymbol:
    n = reduce(operator.mul, grid_size, 1)
    return MatrixSymbol(name, n, n)
