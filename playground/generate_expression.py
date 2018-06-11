from evostencils.generator import ExpressionGenerator
from evostencils.expressions import scalar

grid = (100,)

x = scalar.generate_vector_on_grid('x', grid)
b = scalar.generate_vector_on_grid('b', grid)
A = scalar.generate_matrix_on_grid('A', grid)

generator = ExpressionGenerator(A, x, b)

expr = generator.generate_individual()
print(generator.compile_expression(expr))
