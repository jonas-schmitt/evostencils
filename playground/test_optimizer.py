from evostencils.optimizer import Optimizer
from evostencils.expressions import scalar, block
from evostencils.evaluation.convergence import *
import sympy as sp
import math
import lfa_lab as lfa

fine_grid_size = (1024, 1024)

x = block.generate_vector_on_grid('x', fine_grid_size)
b = block.generate_vector_on_grid('b', fine_grid_size)
A = block.generate_matrix_on_grid('A', fine_grid_size)


fine = lfa.Grid(2, [1.0, 1.0])
# Create a poisson operator.
fine_operator = lfa.gallery.poisson_2d(fine)
coarse_operator = lfa.gallery.poisson_2d(fine.coarse((2, 2)))
evaluator = ConvergenceEvaluator(fine_operator, coarse_operator, fine, fine_grid_size, (2, 2))


def evaluate(individual, generator):
    expression = generator.compile_scalar_expression(individual)
    iteration_matrix = generator.get_iteration_matrix(expression, sp.block_collapse(generator.grid), sp.block_collapse(generator.rhs))
    spectral_radius = evaluator.compute_spectral_radius(iteration_matrix)
    if spectral_radius == 0.0:
        return math.inf,
    else:
        return spectral_radius,




def main():
    smoother_generator = Optimizer(A, x, b, evaluate)
    pop, log, hof = smoother_generator.ea_simple(1000, 20, 0.5, 0.3)
    print(log.stream)
    print(smoother_generator.compile_scalar_expression(hof[0]))
    return pop, log, hof


if __name__ == "__main__":
    main()

