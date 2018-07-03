from evostencils.generator import SmootherGenerator
from evostencils.expressions import scalar, block
import sympy as sp
import lfa_lab as lfa

grid = (100,)

x = block.generate_vector_on_grid('x', grid)
b = block.generate_vector_on_grid('b', grid)
A = block.generate_matrix_on_grid('A', grid)

generator = SmootherGenerator(A, x, b)

individual = generator.generate_individual()
M = sp.block_collapse(generator.compile_expression(individual.tree1))
N = sp.block_collapse(generator.compile_expression(individual.tree2))
expression = M.I * N
print(expression)
from evostencils.evaluation.convergence import *
# Create a 2D grid with step-size (1/32, 1/32).
fine = lfa.Grid(2, [1.0/32, 1.0/32])

# Create a poisson operator.
operator = lfa.gallery.poisson_2d(fine)

evaluator = ConvergenceEvaluator(operator)

smoother = evaluator.transform(expression)
print(smoother)
try:
    symbol = smoother.symbol()
    print(symbol.spectral_radius())
except RuntimeError as re:
    pass

