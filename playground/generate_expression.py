from evostencils.generator import ExpressionGenerator
from evostencils.expressions import scalar, block

grid = (100,)

x = block.generate_vector_on_grid('x', grid)
b = block.generate_vector_on_grid('b', grid)
A = block.generate_matrix_on_grid('A', grid)

generator = ExpressionGenerator(A, x, b)

individual = generator.generate_individual()
#print(generator.compile_expression(individual))
print(generator.compile_expression(individual.tree1))
print(generator.compile_expression(individual.tree2))
