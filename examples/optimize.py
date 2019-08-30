from evostencils.optimization.program import Optimizer
from evostencils.expressions import base
from evostencils.evaluation.convergence import ConvergenceEvaluator
from evostencils.evaluation.roofline import RooflineEvaluator
from evostencils.code_generation.exastencils import ProgramGenerator
from evostencils.initialization import parser
import os
import pickle
import lfa_lab as lfa
import sys


def main():
    compiler_path = f'/home/jonas/Schreibtisch/exastencils/Compiler/compiler.jar'
    base_path = f'/home/jonas/Schreibtisch/exastencils/Examples'
    settings_path = f'BiHarmonic/2D_FD_BiHarmonic_fromL2.settings'
    knowledge_path = f'BiHarmonic/2D_FD_BiHarmonic_fromL2.knowledge'
    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path)
    program_generator.run_exastencils_compiler()
    program_generator.run_c_compiler()
    time, convergence_factor = program_generator.evaluate()
    print(f'Time: {time}, Convergence factor: {convergence_factor}')
    """
    lfa_grids = [lfa.Grid(dimension, sz) for sz in step_sizes]
    convergence_evaluator = ConvergenceEvaluator(lfa_grids, coarsening_factors, dimension)
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
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                          convergence_evaluator=convergence_evaluator,
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
"""

if __name__ == "__main__":
    main()

