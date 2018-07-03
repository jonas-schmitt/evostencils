from evostencils.generator import SmootherGenerator
from evostencils.expressions import scalar, block

grid = (100,)

x = block.generate_vector_on_grid('x', grid)
b = block.generate_vector_on_grid('b', grid)
A = block.generate_matrix_on_grid('A', grid)

generator = SmootherGenerator(A, x, b)

individual = generator.generate_individual()
print(generator.compile_scalar_individual(individual))
