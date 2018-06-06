from evostencils.generator import ExpressionGenerator
import evostencils.matrixtypes as mt
import sympy as sp

grid = (100,)

x = mt.generate_vector_on_grid('x', grid)
b = mt.generate_vector_on_grid('b', grid)
A = mt.generate_matrix_on_grid('A', grid)

generator = ExpressionGenerator(A, x, b)

expr = generator.generate_individual()
print(generator.compile_expression(expr))
