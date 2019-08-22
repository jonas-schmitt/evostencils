from evostencils.optimization.program_system import Optimizer
from evostencils.expressions import multigrid, system
from evostencils.stencils.gallery import *
from evostencils.evaluation.convergence import ConvergenceEvaluatorSystem
from evostencils.evaluation.roofline_system import RooflineEvaluator
from evostencils.code_generation.exastencils import ProgramGenerator
from evostencils.code_generation import parser
from evostencils.initialization.system import generate_operators_from_l2_information
import os
import pickle
import lfa_lab as lfa
import sys


def main():

    exastencils_path = '/home/jonas/Schreibtisch/exastencils'
    if len(sys.argv[1:]) > 0 and os.path.exists(sys.argv[1]):
        exastencils_path = sys.argv[1]
    settings_path = f'{exastencils_path}/Examples/Stokes/2D_FD_Stokes_fromL2.settings'
    knowledge_path = f'{exastencils_path}/Examples/Stokes/2D_FD_Stokes_fromL2.knowledge'
    base_path, config_name = parser.extract_settings_information(settings_path)
    dimensionality, min_level, max_level = parser.extract_knowledge_information(knowledge_path)
    l3_path = f'{exastencils_path}/Examples/Debug/{config_name}_debug.exa3'
    equations, operators, fields = parser.extract_l2_information(l3_path, dimensionality)
    dimension = dimensionality
    size = 2**max_level
    grid_size = (size, size)
    h = 1/(2**max_level)
    step_size = (h, h)
    coarsening_factor = (2, 2)
    step_sizes = [step_size, step_size]
    coarsening_factors = [coarsening_factor, coarsening_factor]

    grid = base.Grid(grid_size, step_size)
    u = system.Approximation('u', [base.ZeroApproximation(grid), base.ZeroApproximation(grid)])
    f = system.RightHandSide('f', [base.RightHandSide('f_0', grid), base.RightHandSide('f_1', grid)])

    problem_name = 'poisson_2D_constant'
    stencil_generator = Poisson2D()
    interpolation_generator = MultilinearInterpolationGenerator(coarsening_factor)
    restriction_generator = FullWeightingRestrictionGenerator(coarsening_factor)

    laplace = base.Operator('laplace', grid, stencil_generator)
    I = base.Identity(grid)
    Z = base.ZeroOperator(grid)

    A = system.Operator('A', [[laplace, I], [Z, laplace]])
    R = system.Restriction('full-weighting restriction', u.grid, system.get_coarse_grid(u.grid, coarsening_factors), restriction_generator)
    P = system.Prolongation('multilinear interpolation', u.grid, system.get_coarse_grid(u.grid, coarsening_factors), interpolation_generator)

    lfa_grids = [lfa.Grid(dimension, sz) for sz in step_sizes]
    convergence_evaluator = ConvergenceEvaluatorSystem(lfa_grids, coarsening_factors, dimension)
    bytes_per_word = 8
    peak_performance = 4 * 16 * 3.6 * 1e9 # 4 Cores * 16 DP FLOPS * 3.6 GHz
    peak_bandwidth = 34.1 * 1e9 # 34.1 GB/s
    runtime_cgs = 1e-7 # example value
    performance_evaluator = RooflineEvaluator(peak_performance, peak_bandwidth, bytes_per_word, runtime_cgs)
    infinity = 1e100
    epsilon = 1e-10
    required_convergence = 0.9

    if not os.path.exists(problem_name):
        os.makedirs(problem_name)
    checkpoint_directory_path = f'{problem_name}/checkpoints'
    if not os.path.exists(checkpoint_directory_path):
        os.makedirs(checkpoint_directory_path)
    optimizer = Optimizer(A, u, f, dimension, coarsening_factors, P, R, min_level, max_level, convergence_evaluator=convergence_evaluator,
                          performance_evaluator=performance_evaluator,
                          epsilon=epsilon, infinity=infinity, checkpoint_directory_path=checkpoint_directory_path)
    # restart_from_checkpoint = True
    restart_from_checkpoint = False
    # program, pops, stats = optimizer.default_optimization(es_lambda=10, es_generations=3,
    #                                                       restart_from_checkpoint=restart_from_checkpoint)
    _, pops, stats = optimizer.default_optimization(gp_mu=100, gp_lambda=100, gp_generations=20, es_generations=20,
                                                    required_convergence=required_convergence,
                                                    restart_from_checkpoint=restart_from_checkpoint)
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

