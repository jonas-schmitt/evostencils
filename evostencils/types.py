from sympy import MatrixSymbol
from sympy import BlockMatrix
import sympy as sp
from functools import reduce
import operator

class MatrixTypeMetaClass(type):
    def __new__(mcs, class_name, bases, dct):
        return super(MatrixTypeMetaClass, mcs).__new__(mcs, class_name, bases, dct)

    def __eq__(self, other):
        return self.shape == other.shape \
               and self.diagonal == other.diagonal \
               and self.lower_triangle == other.lower_triangle \
               and self.upper_triangle == other.upper_triangle

    def __subclasscheck__(self, other):
        is_subclass = True
        if self.shape != other.shape:
            return False
        elif not self.diagonal:
            is_subclass = is_subclass and not other.diagonal
        elif not self.lower_triangle:
            is_subclass = is_subclass and not other.lower_triangle
        elif not self.upper_triangle:
            is_subclass = is_subclass and not other.upper_triangle
        return is_subclass

    def __hash__(self):
        return hash((*self.shape, self.diagonal, self.lower_triangle, self.upper_triangle))


def generate_matrix_type(shape, diag=True, lower_triangle=True, upper_triangle=True):
    return MatrixTypeMetaClass("MatrixType", (), {
        "shape": shape,
        "diagonal": diag,
        "lower_triangle": lower_triangle,
        "upper_triangle": upper_triangle
    })


def generate_diagonal_matrix_type(shape):
    return generate_matrix_type(shape, diag=True, lower_triangle=False, upper_triangle=False)


def generate_strictly_lower_triangular_matrix_type(shape):
    return generate_matrix_type(shape, diag=False, lower_triangle=True, upper_triangle=False)


def generate_strictly_upper_triangular_matrix_type(shape):
    return generate_matrix_type(shape, diag=False, lower_triangle=False, upper_triangle=True)


def generate_lower_triangular_matrix_type(shape):
    return generate_matrix_type(shape, diag=True, lower_triangle=True, upper_triangle=False)


def generate_upper_triangular_matrix_type(shape):
    return generate_matrix_type(shape, diag=True, lower_triangle=False, upper_triangle=True)
