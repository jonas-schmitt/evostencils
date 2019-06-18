from evostencils.optimization.program import Optimizer
from evostencils.expressions import multigrid
from evostencils.stencils.gallery import *
from evostencils.evaluation.convergence import ConvergenceEvaluator
from evostencils.evaluation.roofline import RooflineEvaluator
from evostencils.code_generation.exastencils import ProgramGenerator
import os
from evostencils.code_generation.gallery.finite_differences.poisson_2D import InitializationInformation
# from evostencils.exastencils.gallery.finite_differences.poisson_2D_variable_coefficients \
#    import InitializationInformation
# from evostencils.exastencils.gallery.finite_differences.poisson_3D import InitializationInformation
#from evostencils.code_generation.gallery.finite_differences.poisson_3D_variable_coefficients \
#    import InitializationInformation
import pickle
import lfa_lab as lfa
import sys


def main():
    dimension = 2
    # dimension = 3
    min_level = 2
    max_level = 10
    # min_level = 3
    # max_level = 7
    size = 2**max_level
    grid_size = (size, size)
    # grid_size = (size, size, size)
    h = 1/(2**max_level)
    step_size = (h, h)
    # step_size = (h, h, h)
    coarsening_factor = (2, 2)
    # coarsening_factor = (2, 2, 2)

    grid = base.Grid(grid_size, step_size)
    u = base.Approximation('u', grid)
    b = base.RightHandSide('f', grid)

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

    A = base.Operator('A', grid, stencil_generator)
    I = base.Identity(grid)
    P = multigrid.Prolongation('P', grid, multigrid.get_coarse_grid(grid, coarsening_factor), interpolation_generator)
    R = multigrid.Restriction('R', grid, multigrid.get_coarse_grid(grid, coarsening_factor), restriction_generator)

    lfa_grid = lfa.Grid(dimension, step_size)
    convergence_evaluator = ConvergenceEvaluator(lfa_grid, coarsening_factor, dimension, lfa.gallery.ml_interpolation, lfa.gallery.fw_restriction)
    infinity = 1e100
    epsilon = 1e-10
    required_convergence = 0.9

    bytes_per_word = 8
    peak_performance = 4 * 16 * 3.6 * 1e9 # 4 Cores * 16 DP FLOPS * 3.6 GHz
    peak_bandwidth = 34.1 * 1e9 # 34.1 GB/s
    runtime_cgs = 1e-3 # example value
    performance_evaluator = RooflineEvaluator(peak_performance, peak_bandwidth, bytes_per_word, runtime_cgs)
    # pass path to exa
    exastencils_path = ''
    if len(sys.argv[1:]) > 0 and os.path.exists(sys.argv[1]):
        exastencils_path = sys.argv[1]
    program_generator = ProgramGenerator(problem_name, exastencils_path, A, u, b, I, P, R,
                                         dimension, coarsening_factor, min_level, max_level,
                                         initialization_information=InitializationInformation)

    if not os.path.exists(problem_name):
        os.makedirs(problem_name)
    checkpoint_directory_path = f'{problem_name}/checkpoints'
    if not os.path.exists(checkpoint_directory_path):
        os.makedirs(checkpoint_directory_path)
    optimizer = Optimizer(A, u, b, dimension, coarsening_factor, P, R, min_level, max_level, convergence_evaluator=convergence_evaluator,
                          performance_evaluator=performance_evaluator, program_generator=program_generator,
                          epsilon=epsilon, infinity=infinity, checkpoint_directory_path=checkpoint_directory_path)
    restart_from_checkpoint = True
    # restart_from_checkpoint = False
    # program, pops, stats = optimizer.default_optimization(es_lambda=10, es_generations=3,
    #                                                       restart_from_checkpoint=restart_from_checkpoint)
    program, pops, stats = optimizer.default_optimization(gp_mu=20, gp_lambda=20, gp_generations=10,
                                                          es_generations=5, required_convergence=required_convergence,
                                                          restart_from_checkpoint=restart_from_checkpoint)
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

