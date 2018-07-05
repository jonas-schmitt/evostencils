from evostencils.optimizer import SmootherOptimizer
from evostencils.expressions import scalar, block
from evostencils.evaluation.convergence import *
import sympy as sp
import math
import lfa_lab as lfa

grid = (1000, 10000)

x = block.generate_vector_on_grid('x', grid)
b = block.generate_vector_on_grid('b', grid)
A = block.generate_matrix_on_grid('A', grid)


# Create a 2D grid with step-size (1/32, 1/32).
fine = lfa.Grid(2, [1.0, 1.0])
# Create a poisson operator.
operator = lfa.gallery.poisson_2d(fine)
evaluator = ConvergenceEvaluator(operator)


def evaluate(individual, generator):
    expression = generator.compile_scalar_expression(individual)
    iteration_matrix = generator.get_iteration_matrix(expression, sp.block_collapse(generator.grid), sp.block_collapse(generator.rhs))
    spectral_radius = evaluator.compute_spectral_radius(iteration_matrix)
    if spectral_radius == 0.0:
        return math.inf,
    else:
        return spectral_radius,


smoother_generator = SmootherOptimizer(A, x, b, evaluate)
result = smoother_generator.optimize()

