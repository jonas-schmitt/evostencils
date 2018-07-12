from evostencils.optimizer import Optimizer
from evostencils.expressions import scalar, block, transformations, multigrid
from evostencils.evaluation.convergence import *
import sympy as sp
import math
import lfa_lab as lfa

infinity = 1e10
fine_grid_size = (8, 8)

x = block.generate_vector_on_grid('x', fine_grid_size)
b = block.generate_vector_on_grid('b', fine_grid_size)
A = block.generate_matrix_on_grid('A', fine_grid_size)


fine = lfa.Grid(2, [1.0, 1.0])
# Create a poisson operator.
fine_operator = lfa.gallery.poisson_2d(fine)
coarse_operator = lfa.gallery.poisson_2d(fine.coarse((2, 2)))
evaluator = ConvergenceEvaluator(fine_operator, coarse_operator, fine, fine_grid_size, (2, 2))


def evaluate(individual, generator):
    expression = transformations.fold_intergrid_operations(generator.compile_scalar_expression(individual))
    iteration_matrix = generator.get_iteration_matrix(expression, sp.block_collapse(generator.grid), sp.block_collapse(generator.rhs))
    spectral_radius = evaluator.compute_spectral_radius(iteration_matrix)
    atoms = expression.atoms(sp.MatMul, sp.MatAdd)
    expr_length = len(atoms)
    if spectral_radius == 0.0:
        spectral_radius = infinity
    if expr_length <= 1:
        expr_length = infinity
    return spectral_radius, expr_length


def main():
    smoother_generator = Optimizer(A, x, b, evaluate)
    pop, log, hof = smoother_generator.ea_simple(200, 20, 0.5, 0.3)
    #pop, log, hof = smoother_generator.ea_mu_plus_lambda(1000, 20, 0.5, 0.3, 500, 500)
    print(smoother_generator.compile_scalar_expression(hof[0]))
    return pop, log, hof


if __name__ == "__main__":
    main()

