from evostencils.optimization.system import Optimizer
from evostencils.expressions import multigrid, system
from evostencils.stencils.gallery import *
from evostencils.evaluation.convergence import ConvergenceEvaluatorSystem
from evostencils.code_generation.exastencils import ProgramGenerator
import os
import pickle
import lfa_lab as lfa
import sys


def main():
    dimension = 2
    min_level = 2
    max_level = 10
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
    infinity = 1e100
    epsilon = 1e-10
    required_convergence = 0.9

    if not os.path.exists(problem_name):
        os.makedirs(problem_name)
    checkpoint_directory_path = f'{problem_name}/checkpoints'
    if not os.path.exists(checkpoint_directory_path):
        os.makedirs(checkpoint_directory_path)
    optimizer = Optimizer(A, u, f, dimension, coarsening_factors, P, R, min_level, max_level, convergence_evaluator=convergence_evaluator,
                          epsilon=epsilon, infinity=infinity, checkpoint_directory_path=checkpoint_directory_path)
    # restart_from_checkpoint = True
    restart_from_checkpoint = False
    # program, pops, stats = optimizer.default_optimization(es_lambda=10, es_generations=3,
    #                                                       restart_from_checkpoint=restart_from_checkpoint)
    _, pops, stats = optimizer.default_optimization(gp_mu=20, gp_lambda=20, gp_generations=20, es_generations=20,
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

