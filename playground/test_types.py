from evostencils.matrixtypes import *
from sympy import MatrixSymbol
grid = (2,)

A = generate_matrix_on_grid('A', grid)

MatrixType1 = generate_matrix_type(A.shape)
MatrixType2 = generate_matrix_type(A.shape)
print(MatrixType1 == MatrixType2)
print(isinstance(MatrixType1, MatrixType2))
print(issubclass(MatrixType1, MatrixType2))
