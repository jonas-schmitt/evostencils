from evostencils.optimizer import Optimizer
from evostencils.expressions import base, transformations
from evostencils.stencils.gallery import *
from evostencils.evaluation.convergence import ConvergenceEvaluator
#from evostencils.evaluation.roofline import *
import lfa_lab as lfa


def main():
    # Create a 2D grid with step-size (1/32, 1/32).
    dimension = 2
    grid_size = (512, 512)
    step_size = (1.0, 1.0)
    coarsening_factor = (2, 2)

    lfa_grid = lfa.Grid(dimension, step_size)
    lfa_operator = lfa.gallery.poisson_2d(lfa_grid)
    lfa_coarse_operator = lfa.gallery.poisson_2d(lfa_grid.coarse(coarsening_factor))

    u = base.generate_grid('u', grid_size, step_size)
    b = base.generate_grid('f', grid_size, step_size)
    A = base.generate_operator_on_grid('A', u, generate_poisson_2d)

    convergence_evaluator = ConvergenceEvaluator(lfa_grid, coarsening_factor, dimension, lfa.gallery.poisson_2d, lfa.gallery.ml_interpolation, lfa.gallery.fw_restriction)
    infinity = 1e10
    epsilon = 1e-9

    #bytes_per_word = 8
    #peak_performance = 4 * 16 * 3.6 * 1e9 # 4 Cores * 16 DP FLOPS * 3.6 GHz
    #peak_bandwidth = 34.1 * 1e9 # 34.1 GB/s
    #performance_evaluator = RooflineEvaluator(peak_performance, peak_bandwidth, bytes_per_word)

    optimizer = Optimizer(A, u, b, dimension, coarsening_factor, convergence_evaluator=convergence_evaluator,
                          performance_evaluator=None, epsilon=epsilon, infinity=infinity)
    pop, log, hof = optimizer.default_optimization(200, 15, 0.5, 0.3)

    i = 1
    print('\n')
    for ind in hof:
        print(f'Individual {i} with fitness {ind.fitness}')
        expression = transformations.fold_intergrid_operations(optimizer.compile_expression(ind))
        expression = transformations.set_weights(expression, ind.weights)
        print(f'Update expression: {repr(expression)}')
        iteration_matrix = optimizer.get_iteration_matrix(expression, optimizer.grid, optimizer.rhs)
        print(f'Iteration Matrix: {repr(iteration_matrix)}\n')
        try:
            optimizer.visualize_tree(ind, f'tree{i}')
        except:
            pass
        i = i + 1
    return pop, log, hof


if __name__ == "__main__":
    main()

