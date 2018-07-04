from evostencils.optimizer import SmootherOptimizer
from evostencils.expressions import scalar, block
from evostencils.expressions import transformations as tf
import sympy as sp
import lfa_lab as lfa

grid = (100,)

x = block.generate_vector_on_grid('x', grid)
b = block.generate_vector_on_grid('b', grid)
A = block.generate_matrix_on_grid('A', grid)

generator = SmootherOptimizer(A, x, b)

individual = generator.generate_individual()
expression = generator.compile_scalar_expression(individual)
print(expression)
iteration_matrix = generator.get_iteration_matrix(expression, sp.block_collapse(generator.grid), sp.block_collapse(generator.rhs))
from evostencils.evaluation.convergence import *
# Create a 2D grid with step-size (1/32, 1/32).
fine = lfa.Grid(2, [1.0/1000, 1.0/1000])

# Create a poisson operator.
operator = lfa.gallery.poisson_2d(fine)

evaluator = ConvergenceEvaluator(operator)
#iteration_matrix = sp.block_collapse(sp.Identity(A.shape[0]) - generator._diagonal.I * generator.operator * sp.Identity(A.shape[0]))
print(iteration_matrix)
smoother = evaluator.transform(iteration_matrix)
print(smoother)
try:
    symbol = smoother.symbol()
    print(symbol.spectral_radius())
except RuntimeError as re:
    pass

population = generator._toolbox.population(100)

