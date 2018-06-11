from evostencils.types import *
from evostencils.expressions import scalar
from evostencils.expressions import block

from sympy import BlockMatrix

grid = (2,)

A = scalar.generate_matrix_on_grid('A', grid)

MatrixType1 = generate_matrix_type(A.shape)
MatrixType2 = generate_matrix_type(A.shape)
print(MatrixType1 == MatrixType2)
print(isinstance(MatrixType1, MatrixType2))
print(issubclass(MatrixType1, MatrixType2))

B = BlockMatrix([[A, A], [A, A]])
B_d = block.get_block_diagonal(B)
print(B_d)
#print(B.blockshape)
