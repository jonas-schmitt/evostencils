from evostencils.types.matrix import *
from evostencils.expressions import base

grid_size = (2,)

A = base.generate_operator('A', grid_size)

MatrixType1 = generate_diagonal_matrix_type(A.shape)
MatrixType2 = generate_lower_triangular_matrix_type(A.shape)
print(MatrixType1 == MatrixType2)
print(isinstance(MatrixType1, MatrixType2))
#print(issubclass(MatrixType1, MatrixType2))
OperatorType = generate_matrix_type(A.shape)
BlockDiagonalOperatorType = generate_block_diagonal_matrix_type(A.shape)
DiagonalOperatorType = generate_diagonal_matrix_type(A.shape)
print(issubclass(DiagonalOperatorType, BlockDiagonalOperatorType))
