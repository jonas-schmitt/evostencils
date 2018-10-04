from evostencils.optimizer import Optimizer
from evostencils.expressions import transformations
from evostencils.stencils.constant import Stencil
from evostencils.evaluation.convergence import *
from evostencils.evaluation.roofline import *
import lfa_lab as lfa


def main():
    infinity = 1e10
    epsilon = 1e-9
    dimension = 2
    fine_grid_size = (1000, 1000)
    operator_stencil_entries = [
        (( 0, -1), -1.0),
        ((-1,  0), -1.0),
        (( 0,  0),  4.0),
        (( 1,  0), -1.0),
        (( 0,  1), -1.0)
    ]
    x = base.generate_grid('x', fine_grid_size)
    b = base.generate_grid('b', fine_grid_size)
    A = base.generate_operator_on_grid('A', fine_grid_size, Stencil(operator_stencil_entries))

    coarsening_factor = (2, 2)
    fine = lfa.Grid(dimension, [1.0, 1.0])
    fine_operator = lfa.gallery.poisson_2d(fine)
    coarse_operator = lfa.gallery.poisson_2d(fine.coarse(coarsening_factor))
    convergence_evaluator = ConvergenceEvaluator(coarse_operator, fine, fine_grid_size, coarsening_factor,,

    bytes_per_word = 8
    peak_performance = 4 * 16 * 3.6 * 1e9 # 4 Cores * 16 DP FLOPS * 3.6 GHz
    peak_bandwidth = 34.1 * 1e9 # 34.1 GB/s
    performance_evaluator = RooflineEvaluator(peak_performance, peak_bandwidth, bytes_per_word)

    optimizer = Optimizer(A, x, b, dimension, coarsening_factor, convergence_evaluator=convergence_evaluator,
                          performance_evaluator=performance_evaluator, epsilon=epsilon, infinity=infinity)
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

