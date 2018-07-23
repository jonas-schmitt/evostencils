from sympy import MatrixSymbol, BlockMatrix, ZeroMatrix
from evostencils.expressions import scalar
from functools import reduce
import operator


def map_block_matrix(fun, B: BlockMatrix) -> BlockMatrix:
    result_blocks = []
    for i in range(0, B.blocks.rows):
        result_blocks.append([])
        for j in range(0, B.blocks.cols):
            result_blocks[-1].append(fun(B.blocks[i * B.blocks.rows + j], (i, j)))
    return BlockMatrix(result_blocks)


def get_diagonal(B: BlockMatrix) -> BlockMatrix:
    if B.blockshape[0] < 2 and B.blockshape[1] < 2:
        def filter_matrix(matrix, index):
            return scalar.get_diagonal(matrix)
    else:
        def filter_matrix(matrix, index):
            if index[0] == index[1]:
                return scalar.get_diagonal(matrix)
            else:
                return ZeroMatrix(*matrix.shape)
    return map_block_matrix(filter_matrix, B)


def get_lower_triangle(B: BlockMatrix) -> BlockMatrix:
    if B.blockshape[0] < 2 and B.blockshape[1] < 2:
        def filter_matrix(matrix, index):
            return scalar.get_lower_triangle(matrix)
    else:
        def filter_matrix(matrix, index):
            if index[0] == index[1]:
                return scalar.get_lower_triangle(matrix)
            elif index[0] > index[1]:
                return matrix
            else:
                return ZeroMatrix(*matrix.shape)
    return map_block_matrix(filter_matrix, B)


def get_upper_triangle(B: BlockMatrix) -> BlockMatrix:
    if B.blockshape[0] < 2 and B.blockshape[1] < 2:
        def filter_matrix(matrix, index):
            return scalar.get_upper_triangle(matrix)
    else:
        def filter_matrix(matrix, index):
            if index[0] == index[1]:
                return scalar.get_upper_triangle(matrix)
            elif index[0] < index[1]:
                return matrix
            else:
                return ZeroMatrix(*matrix.shape)
    return map_block_matrix(filter_matrix, B)


def get_block_diagonal(B: BlockMatrix) -> BlockMatrix:
    def filter_matrix(matrix, index):
            if index[0] == index[1]:
                return matrix
            else:
                return ZeroMatrix(*matrix.shape)
    return map_block_matrix(filter_matrix, B)


def get_block_lower_triangle(B: BlockMatrix) -> BlockMatrix:
    def filter_matrix(matrix, index):
        if index[0] > index[1]:
            return matrix
        else:
            return ZeroMatrix(*matrix.shape)
    return map_block_matrix(filter_matrix, B)


def get_block_upper_triangle(B: BlockMatrix) -> BlockMatrix:
    def filter_matrix(matrix, index):
        if index[0] < index[1]:
            return matrix
        else:
            return ZeroMatrix(*matrix.shape)
    return map_block_matrix(filter_matrix, B)


def generate_vector_on_grid(name: str, grid_size: tuple) -> MatrixSymbol:
    n = reduce(operator.mul, grid_size, 1)
    x = BlockMatrix([[MatrixSymbol(name, n, 1)]])
    return BlockMatrix([[x]])


def generate_matrix_on_grid(name: str, grid_size: tuple) -> MatrixSymbol:
    n = reduce(operator.mul, grid_size, 1)
    A = MatrixSymbol(name, n, n)
    return BlockMatrix([[A]])
