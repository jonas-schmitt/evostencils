from evostencils.optimizer import Optimizer, load_checkpoint_from_file
from evostencils.expressions import multigrid
from evostencils.stencils.gallery import *
from evostencils.evaluation.convergence import ConvergenceEvaluator
from evostencils.evaluation.roofline import RooflineEvaluator
from evostencils.exastencils.generation import ProgramGenerator
import os
from evostencils.exastencils.gallery.finite_differences.poisson_2D import InitializationInformation
# from evostencils.exastencils.gallery.finite_differences.poisson_2D_variable_coefficients \
#    import InitializationInformation
# from evostencils.exastencils.gallery.finite_differences.poisson_3D import InitializationInformation
# from evostencils.exastencils.gallery.finite_differences.poisson_3D_variable_coefficients \
#    import InitializationInformation
import pickle
import lfa_lab as lfa


def main():
    dimension = 2
    # dimension = 3
    levels = 8
    max_levels = 8
    # levels = 4
    # max_levels = 6
    size = 2**max_levels
    grid_size = (size, size)
    # grid_size = (size, size, size)
    h = 1/(2**max_levels)
    step_size = (h, h)
    # step_size = (h, h, h)
    coarsening_factor = (2, 2)
    # coarsening_factor = (2, 2, 2)

    u = base.generate_grid('u', grid_size, step_size)
    b = base.generate_rhs('f', grid_size, step_size)

    problem_name = 'poisson_2D_constant'
    # problem_name = 'poisson_2D_variable'
    # problem_name = 'poisson_3D_constant'
    # problem_name = 'poisson_3D_variable'
    stencil_generator = Poisson2D()
    # stencil_generator = Poisson2DVariableCoefficients(get_coefficient_2D, (0.5, 0.5))
    # stencil_generator = Poisson3D()
    # stencil_generator = Poisson3DVariableCoefficients(get_coefficient_3D, (0.5, 0.5, 0.5))
    interpolation_generator = InterpolationGenerator(coarsening_factor)
    restriction_generator = RestrictionGenerator(coarsening_factor)

    A = base.generate_operator_on_grid('A', u, stencil_generator)
    I = base.Identity(A.shape, u)
    P = multigrid.get_interpolation(u, multigrid.get_coarse_grid(u, coarsening_factor), interpolation_generator)
    R = multigrid.get_restriction(u, multigrid.get_coarse_grid(u, coarsening_factor), restriction_generator)

    lfa_grid = lfa.Grid(dimension, step_size)
    convergence_evaluator = ConvergenceEvaluator(lfa_grid, coarsening_factor, dimension, lfa.gallery.ml_interpolation, lfa.gallery.fw_restriction)
    infinity = 1e100
    epsilon = 1e-20

    bytes_per_word = 8
    peak_performance = 4 * 16 * 3.6 * 1e9 # 4 Cores * 16 DP FLOPS * 3.6 GHz
    peak_bandwidth = 34.1 * 1e9 # 34.1 GB/s
    runtime_cgs = 1e-3 # example value
    performance_evaluator = RooflineEvaluator(peak_performance, peak_bandwidth, bytes_per_word, runtime_cgs)
    exastencils_path = ''
    program_generator = ProgramGenerator(problem_name, exastencils_path, A, u, b, I, P, R,
                                         dimension, coarsening_factor,
                                         initialization_information=InitializationInformation)

    if not os.path.exists(problem_name):
        os.makedirs(problem_name)
    checkpoint_directory_path = f'{problem_name}/checkpoints'
    if not os.path.exists(checkpoint_directory_path):
        os.makedirs(checkpoint_directory_path)
    optimizer = Optimizer(A, u, b, dimension, coarsening_factor, P, R, levels, convergence_evaluator=convergence_evaluator,
                          performance_evaluator=performance_evaluator, program_generator=program_generator,
                          epsilon=epsilon, infinity=infinity, checkpoint_directory_path=checkpoint_directory_path)
    restart_from_checkpoint = True
    # program, pops, stats = optimizer.default_optimization()
    program, pops, stats = optimizer.default_optimization(gp_mu=50, gp_lambda=50, gp_generations=10, es_lambda=20,
                                                          es_generations=20, restart_from_checkpoint=restart_from_checkpoint)
    print(program)
    program_generator.write_program_to_file(program)
    log_dir_name = f'{problem_name}/data'
    if not os.path.exists(log_dir_name):
        os.makedirs(log_dir_name)
    i = 1
    for log in stats:
        pickle.dump(log, open(f"{log_dir_name}/log{i}.p", "wb"))
        i += 1
        # optimizer.plot_average_fitness(log)
        # optimizer.plot_minimum_fitness(log)
    # for pop in pops:
    #    optimizer.plot_pareto_front(pop)


if __name__ == "__main__":
    main()

