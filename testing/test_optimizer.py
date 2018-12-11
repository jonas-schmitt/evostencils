from evostencils.optimizer import Optimizer
from evostencils.expressions import multigrid
from evostencils.stencils.gallery import *
from evostencils.evaluation.convergence import ConvergenceEvaluator
from evostencils.evaluation.roofline import RooflineEvaluator
from evostencils.exastencils.generation import ProgramGenerator
from evostencils.exastencils.gallery.finite_differences.poisson_2D import InitializationInformation
import lfa_lab as lfa


def main():
    dimension = 2
    levels = 8
    size = 2**20
    grid_size = (size, size)
    h = 1/(2**levels)
    step_size = (h, h)
    coarsening_factor = (2, 2)

    u = base.generate_grid('u', grid_size, step_size)
    b = base.generate_rhs('f', grid_size, step_size)

    stencil_generator = Poisson2D()
    interpolation_generator = InterpolationGenerator(coarsening_factor)
    restriction_generator = RestrictionGenerator(coarsening_factor)

    A = base.generate_operator_on_grid('A', u, stencil_generator)
    I = base.Identity(A.shape, u)
    P = multigrid.get_interpolation(u, multigrid.get_coarse_grid(u, coarsening_factor), interpolation_generator)
    R = multigrid.get_restriction(u, multigrid.get_coarse_grid(u, coarsening_factor), restriction_generator)

    lfa_grid = lfa.Grid(dimension, step_size)
    convergence_evaluator = ConvergenceEvaluator(lfa_grid, coarsening_factor, dimension, lfa.gallery.ml_interpolation, lfa.gallery.fw_restriction)
    infinity = 1e20
    epsilon = 1e-10

    bytes_per_word = 8
    peak_performance = 4 * 16 * 3.6 * 1e9 # 4 Cores * 16 DP FLOPS * 3.6 GHz
    peak_bandwidth = 34.1 * 1e9 # 34.1 GB/s
    performance_evaluator = RooflineEvaluator(peak_performance, peak_bandwidth, bytes_per_word)
    program_generator = ProgramGenerator('2D_FD_Poisson', '/local/ja42rica/ScalaExaStencil', A, u, b, I, P, R,
                                         dimension, coarsening_factor,
                                         initialization_information=InitializationInformation)
    optimizer = Optimizer(A, u, b, dimension, coarsening_factor, P, R, levels, convergence_evaluator=convergence_evaluator,
                          performance_evaluator=performance_evaluator, program_generator=program_generator, epsilon=epsilon, infinity=infinity)
    program = optimizer.default_optimization(500, 30, 0.7, 0.3)
    print(program)
    optimizer._program_generator.write_program_to_file(program)
    """
    generator = optimizer._program_generator
    i = 1
    print('\n')
    for ind in hof:
        print(f'Individual {i} with fitness {ind.fitness}')
        expression = optimizer.compile_expression(ind)[0]
        if i == 1:
            program = generator.generate(expression)
            print(program)
            generator.write_program_to_file(program)
            time = generator.execute()
            print(f"Runtime: {time}")
            best_weights, spectral_radius = optimizer.optimize_weights(expression, iterations=100)
            transformations.set_weights(expression, best_weights)
            program = generator.generate(expression)
            print(program)
            generator.write_program_to_file(program)
            time = generator.execute()
            print(f'Improved spectral radius: {spectral_radius}')
            print(f"Runtime: {time}")
        #print(f'Update expression: {repr(expression)}')
        try:
            optimizer.visualize_tree(ind, f'tree{i}')
        except:
            pass
        i = i + 1

    optimizer.plot_minimum_fitness(log)
    return pop, log, hof
    """


if __name__ == "__main__":
    main()

