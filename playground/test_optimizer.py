from evostencils.optimizer import Optimizer
from evostencils.expressions import transformations, multigrid
from evostencils.stencils import Stencil
from evostencils.evaluation.convergence import *
from evostencils.evaluation.roofline import *
import sympy as sp
import lfa_lab as lfa
import math

infinity = 1e10
epsilon = 1e-6
fine_grid_size = (100, 100)
operator_stencil_entries = [
    (( 0, -1), -1.0),
    ((-1,  0), -1.0),
    (( 0,  0),  4.0),
    (( 1,  0), -1.0),
    (( 0,  1), -1.0)
]
x = base.generate_grid('x', fine_grid_size)
b = base.generate_grid('b', fine_grid_size)
A = base.generate_operator('A', fine_grid_size, Stencil(operator_stencil_entries))

fine = lfa.Grid(2, [1.0, 1.0])
# Create a poisson operator.
fine_operator = lfa.gallery.poisson_2d(fine)
coarse_operator = lfa.gallery.poisson_2d(fine.coarse((2, 2)))
convergence_evaluator = ConvergenceEvaluator(fine_operator, coarse_operator, fine, fine_grid_size, (2, 2))
bytes_per_word = 8
peak_performance = 4 * 16 * 3.6 * 1e9 # 4 Cores * 16 DP FLOPS * 3.6 GHz
peak_bandwidth = 34.1 * 1e9 # 34.1 GB/s
performance_evaluator = RooflineEvaluator(bytes_per_word)


def evaluate(individual, generator):
    expression = transformations.fold_intergrid_operations(generator.compile_expression(individual))
    iteration_matrix = generator.get_iteration_matrix(expression, sp.block_collapse(generator.grid), sp.block_collapse(generator.rhs))
    spectral_radius = convergence_evaluator.compute_spectral_radius(iteration_matrix)
    if spectral_radius == 0.0 or spectral_radius > 1.0:
        return infinity,
    elif spectral_radius < 1.0:
        runtime = performance_evaluator.estimate_runtime(expression, peak_performance, peak_bandwidth)
        return math.log(epsilon) / math.log(spectral_radius) * runtime,
    else:
        raise RuntimeError("Spectral radius out of range")


def main():
    optimizer = Optimizer(A, x, b, 2, 4, evaluate)
    pop, log, hof = optimizer.default_optimization(200, 20, 0.5, 0.3)
    optimizer.visualize_tree(hof[0], "tree")
    i = 1
    print('\n')
    for ind in hof:
        print(f'Individual {i} with fitness {ind.fitness}')
        expression = transformations.fold_intergrid_operations(optimizer.compile_expression(ind))
        print(f'Update expression: {expression}')
        iteration_matrix = optimizer.get_iteration_matrix(expression, optimizer.grid, optimizer.rhs)
        print(f'Iteration Matrix: {iteration_matrix}\n')
        i = i + 1
    return pop, log, hof


if __name__ == "__main__":
    main()

