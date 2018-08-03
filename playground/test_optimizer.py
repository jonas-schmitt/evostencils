from evostencils.optimizer import Optimizer
from evostencils.expressions import transformations, multigrid
from evostencils.stencils import Stencil
from evostencils.evaluation.convergence import *
from evostencils.evaluation.roofline import *
import sympy as sp
import lfa_lab as lfa

infinity = 1e10
fine_grid_size = (8, 8)
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
performance_evaluator = RooflineEvaluator(8)


def evaluate(individual, generator):
    expression = transformations.fold_intergrid_operations(generator.compile_expression(individual))
    iteration_matrix = generator.get_iteration_matrix(expression, sp.block_collapse(generator.grid), sp.block_collapse(generator.rhs))
    spectral_radius = convergence_evaluator.compute_spectral_radius(iteration_matrix)
    arithmetic_intensity = 0
    if spectral_radius == 0.0:
        spectral_radius = infinity
    elif spectral_radius < 1.0:
        list_of_metrics = performance_evaluator.estimate_performance_metrics(expression)
        arithmetic_intensity = performance_evaluator.compute_average_arithmetic_intensity(list_of_metrics)
    return spectral_radius, arithmetic_intensity


def main():
    optimizer = Optimizer(A, x, b, 2, 4, evaluate)
    pop, log, hof = optimizer.default_optimization(100, 10, 0.5, 0.3)
    optimizer.visualize_tree(hof[0], "tree")
    i = 1
    print('\n')
    for ind in hof:
        print(f'Individual {i} with fitness {ind.fitness}')
        expression = transformations.fold_intergrid_operations(optimizer.compile_expression(ind))
        print(f'Update expression: {expression}')
        iteration_matrix = optimizer.get_iteration_matrix(expression, sp.block_collapse(optimizer.grid), sp.block_collapse(optimizer.rhs))
        print(f'Iteration Matrix: {iteration_matrix}\n')
        i = i + 1
    return pop, log, hof


if __name__ == "__main__":
    main()

