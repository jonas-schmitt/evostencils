from evostencils.types.matrix import *
from evostencils.types.multiple import *
from evostencils.expressions import base

grid_size = (2,)
u = base.generate_grid('u', (64,), (1,))
A = base.generate_operator_on_grid('A', u, None)

MatrixType1 = generate_diagonal_matrix_type(A.shape)
MatrixType2 = generate_lower_triangular_matrix_type(A.shape)
print(MatrixType1 == MatrixType2)
print(isinstance(MatrixType1, MatrixType2))
#print(issubclass(MatrixType1, MatrixType2))
OperatorType = generate_matrix_type(A.shape)
BlockDiagonalOperatorType = generate_block_diagonal_matrix_type(A.shape)
DiagonalOperatorType = generate_diagonal_matrix_type(A.shape)
print(issubclass(DiagonalOperatorType, BlockDiagonalOperatorType))
A = generate_new_type('A')
B = generate_new_type('B')
print(issubclass(A, B))
print(hash(A))
print(hash(B))
