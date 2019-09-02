from evostencils.optimization.program import Optimizer
from evostencils.evaluation.convergence import ConvergenceEvaluator
from evostencils.evaluation.performance import PerformanceEvaluator
from evostencils.code_generation.exastencils import ProgramGenerator
import os
import pickle
import lfa_lab


def main():

    # TODO adapt to actual path to exastencils project

    cwd = os.getcwd()
    compiler_path = f'{cwd}/../exastencils/Compiler/compiler.jar'
    base_path = f'{cwd}/../exastencils/Examples'

    # 2D Finite difference discretized Poisson
    settings_path = f'Poisson/2D_FD_Poisson_fromL2.settings'
    knowledge_path = f'Poisson/2D_FD_Poisson_fromL2.knowledge'

    # 3D Finite difference discretized Poisson
    # settings_path = f'Poisson/3D_FD_Poisson_fromL2.settings'
    # knowledge_path = f'Poisson/3D_FD_Poisson_fromL2.knowledge'

    # 2D Finite volume discretized Poisson
    # settings_path = f'Poisson/2D_FV_Poisson_fromL2.settings'
    # knowledge_path = f'Poisson/2D_FV_Poisson_fromL2.knowledge'

    # 3D Finite volume discretized Poisson
    # settings_path = f'Poisson/3D_FV_Poisson_fromL2.settings'
    # knowledge_path = f'Poisson/3D_FV_Poisson_fromL2.knowledge'

    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path)

    # Evaluate baseline program
    # program_generator.run_exastencils_compiler()
    # program_generator.run_c_compiler()
    # time, convergence_factor = program_generator.evaluate()
    # print(f'Time: {time}, Convergence factor: {convergence_factor}')

    # Obtain extracted information from program generator
    dimension = program_generator.dimension
    finest_grid = program_generator.finest_grid
    coarsening_factors = program_generator.coarsening_factor
    min_level = program_generator.min_level
    max_level = program_generator.max_level
    equations = program_generator.equations
    operators = program_generator.operators
    fields = program_generator.fields

    lfa_grids = [lfa_lab.Grid(dimension, g.step_size) for g in finest_grid]
    convergence_evaluator = ConvergenceEvaluator(dimension, coarsening_factors, lfa_grids)
    bytes_per_word = 8
    peak_performance = 4 * 16 * 3.6 * 1e9 # 4 Cores * 16 DP FLOPS * 3.6 GHz
    peak_bandwidth = 34.1 * 1e9 # 34.1 GB/s
    runtime_cgs = 1e-7 # example value
    performance_evaluator = PerformanceEvaluator(peak_performance, peak_bandwidth, bytes_per_word, runtime_cgs)
    infinity = 1e100
    epsilon = 1e-10
    required_convergence = 0.9
    problem_name = program_generator.problem_name

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
    pops, stats = optimizer.default_optimization(gp_mu=1000, gp_lambda=1000, gp_generations=100, es_generations=20,
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

