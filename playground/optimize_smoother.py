from evostencils.optimizer import SmootherOptimizer
from evostencils.expressions import scalar, block
from evostencils.evaluation.convergence import *
import sympy as sp
import math
import lfa_lab as lfa

grid = (1000, 1000)

x = block.generate_vector_on_grid('x', grid)
b = block.generate_vector_on_grid('b', grid)
A = block.generate_matrix_on_grid('A', grid)


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




def main():
    smoother_generator = SmootherOptimizer(A, x, b, evaluate)
    pop, log, hof = smoother_generator.optimize(20, 20)
    print(log.stream)
    print(smoother_generator.compile_scalar_expression(hof[0]))
    return pop, log, hof


if __name__ == "__main__":
    main()

