from sympy import MatrixSymbol
from sympy import BlockMatrix
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
    D = DiagonalMatrixSymbol(f"{A.name}_diagonal", *A.shape)
    #D = DiagonalMatrixSymbol(f"D", *A.shape)
    D.set_source_matrix(A)
    return D


def get_lower_triangle(A) -> SparseMatrixSymbol:
    L = LowerTriangularMatrixSymbol(f"{A.name}_lower", *A.shape)
    #L = LowerTriangularMatrixSymbol(f"L", *A.shape)
    L.set_source_matrix(A)
    return L


def get_upper_triangle(A) -> SparseMatrixSymbol:
    U = UpperTriangularMatrixSymbol(f"{A.name}_upper", *A.shape)
    #U = UpperTriangularMatrixSymbol(f"U", *A.shape)
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


def map_block_matrix(fun, B: BlockMatrix) -> BlockMatrix:
    result_blocks = []
    for i in range(0, B.blocks.rows):
        result_blocks.append([])
        for j in range(0, B.blocks.cols):
            result_blocks[-1].append(fun(B.blocks[i * B.blocks.rows + j], (i, j)))
    return BlockMatrix(result_blocks)


def get_diagonal_from_block_matrix(B: BlockMatrix) -> BlockMatrix:
    def filter(matrix, index):
        if index[0] == index[1]:
            return get_diagonal(matrix)
        else:
            return matrix
    return map_block_matrix(filter, B)





