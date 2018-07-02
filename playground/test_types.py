from evostencils.types import *
from evostencils.expressions import scalar
from evostencils.expressions import block

from sympy import BlockMatrix, Identity

grid = (2,)

A = scalar.generate_matrix_on_grid('A', grid)

MatrixType1 = generate_diagonal_matrix_type(A.shape)
MatrixType2 = generate_lower_triangular_matrix_type(A.shape)
print(MatrixType1 == MatrixType2)
print(isinstance(MatrixType1, MatrixType2))
print(issubclass(MatrixType1, MatrixType2))

B = BlockMatrix([[A, A], [A, A]])
I = Identity(B.shape[0])
print(B * I)
B_d = block.get_block_diagonal(B)
print(B_d)
#print(B.blockshape)
