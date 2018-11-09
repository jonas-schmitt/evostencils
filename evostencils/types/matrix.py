from sympy import MatrixSymbol
from sympy import BlockMatrix
import sympy as sp
from functools import reduce
import operator


class MatrixTypeMetaClass(type):
    def __new__(mcs, class_name, bases, dct):
        return super(MatrixTypeMetaClass, mcs).__new__(mcs, class_name, bases, dct)

    def __eq__(self, other):
        if hasattr(other, 'shape') and hasattr(other, 'diagonal') and hasattr(other, 'block_diagonal') and \
                hasattr(other, 'lower_triangle') \
                and hasattr(other, 'upper_triangle'):
            return self.shape == other.shape \
                   and self.diagonal == other.diagonal \
                   and self.block_diagonal == other.block_diagonal \
                   and self.lower_triangle == other.lower_triangle \
                   and self.upper_triangle == other.upper_triangle
        else:
            return False

    def __subclasscheck__(self, other):
        if hasattr(other, 'shape') and hasattr(other, 'diagonal') and hasattr(other, 'block_diagonal') \
                and hasattr(other, 'lower_triangle') \
                and hasattr(other, 'upper_triangle'):
            is_subclass = True
            if self.shape != other.shape:
                return False
            if not self.diagonal:
                is_subclass = is_subclass and not other.diagonal
            if not self.lower_triangle:
                is_subclass = is_subclass and not other.lower_triangle
            if not self.upper_triangle:
                is_subclass = is_subclass and not other.upper_triangle
            if not self.block_diagonal:
                is_subclass = is_subclass and not other.block_diagonal
            return is_subclass
        else:
            return False

    def __hash__(self):
        return hash((*self.shape, self.diagonal, self.lower_triangle, self.upper_triangle))


def generate_matrix_type(shape, diag=True, block_diagonal=True, lower_triangle=True, upper_triangle=True):
    return MatrixTypeMetaClass("MatrixType", (), {
        "shape": shape,
        "diagonal": diag,
        "block_diagonal": block_diagonal,
        "lower_triangle": lower_triangle,
        "upper_triangle": upper_triangle
    })


def generate_diagonal_matrix_type(shape):
    return generate_matrix_type(shape, diag=True, block_diagonal=False, lower_triangle=False, upper_triangle=False)


def generate_block_diagonal_matrix_type(shape):
    return generate_matrix_type(shape, diag=True, block_diagonal=True, lower_triangle=False, upper_triangle=False)


def generate_strictly_lower_triangular_matrix_type(shape):
    return generate_matrix_type(shape, diag=False, block_diagonal=False, lower_triangle=True, upper_triangle=False)


def generate_strictly_upper_triangular_matrix_type(shape):
    return generate_matrix_type(shape, diag=False, block_diagonal=False, lower_triangle=False, upper_triangle=True)


def generate_lower_triangular_matrix_type(shape):
    return generate_matrix_type(shape, diag=True, block_diagonal=False, lower_triangle=True, upper_triangle=False)


def generate_upper_triangular_matrix_type(shape):
    return generate_matrix_type(shape, diag=True, block_diagonal=False, lower_triangle=False, upper_triangle=True)


def generate_zero_matrix_type(shape):
    return generate_matrix_type(shape, diag=False, block_diagonal=False, lower_triangle=False, upper_triangle=False)


class SolverTypeMetaClass(type):
    def __new__(mcs, class_name, bases, dct):
        return super(SolverTypeMetaClass, mcs).__new__(mcs, class_name, bases, dct)

    def __eq__(self, other):
        if hasattr(other, 'shape') and hasattr(other, 'is_solver'):
            return self.shape == other.shape and self.is_solver == other.is_solver
        else:
            return False

    def __subclasscheck__(self, other):
        if hasattr(other, 'shape') and hasattr(other, 'is_solver'):
            return self.shape == other.shape and self.is_solver == other.is_solver
        else:
            return False

    # TODO be careful. Could collide with other types!
    def __hash__(self):
        return hash((*self.shape, self.is_solver))


def generate_solver_type(shape):
    return SolverTypeMetaClass("SolverType", (), {
        "shape": shape,
        "is_solver": True
    })
