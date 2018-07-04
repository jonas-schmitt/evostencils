from evostencils.types import *
from evostencils.expressions import scalar
from evostencils.expressions import block

from sympy import BlockMatrix, Identity
import sympy as sp

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
from evostencils.expressions import transformations as tf

A = sp.MatrixSymbol('A', 10, 10)
x = sp.MatrixSymbol('x', 10, 1)
b = sp.MatrixSymbol('b', 10, 1)
zeromatrix = sp.ZeroMatrix(10, 10)
zerovector = sp.ZeroMatrix(10, 1)
expression = A * A * zeromatrix * b + (A + zeromatrix) * A * zerovector + zerovector + (A + zeromatrix.T) * x
print(expression)
print(tf.propagate_zero(expression))
