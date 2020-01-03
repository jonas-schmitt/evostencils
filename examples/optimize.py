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
    compiler_path = f'{cwd}/../exastencils/Compiler/Compiler.jar'
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

    # 2D Finite difference discretized Bi-Harmonic Equation
    # settings_path = f'BiHarmonic/2D_FD_BiHarmonic_fromL2.settings'
    # knowledge_path = f'BiHarmonic/2D_FD_BiHarmonic_fromL2.knowledge'

    # 2D Finite volume discretized Stokes
    # settings_path = f'Stokes/2D_FV_Stokes_fromL2.settings'
    # knowledge_path = f'Stokes/2D_FV_Stokes_fromL2.knowledge'

    # 2D Finite difference discretized linear elasticity
    # settings_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.settings'
    # knowledge_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.knowledge'

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
    peak_performance = 20344.07 * 1e6
    peak_bandwidth = 19255.70 * 1e6
    performance_evaluator = PerformanceEvaluator(peak_performance, peak_bandwidth, bytes_per_word)
    infinity = 1e300
    epsilon = 1e-10
    problem_name = program_generator.problem_name

    if not os.path.exists(problem_name):
        os.makedirs(problem_name)
    checkpoint_directory_path = f'{problem_name}/checkpoints'
    if not os.path.exists(checkpoint_directory_path):
        os.makedirs(checkpoint_directory_path)
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                          convergence_evaluator=convergence_evaluator,
                          performance_evaluator=performance_evaluator, program_generator=program_generator,
                          epsilon=epsilon, infinity=infinity, checkpoint_directory_path=checkpoint_directory_path)

    # restart_from_checkpoint = True
    restart_from_checkpoint = False
    levels_per_run = 2
    required_convergence = 0.2
    maximum_block_size = 3
    # Use single-objective optimization
    program, pops, stats = optimizer.evolutionary_optimization(optimization_method=optimizer.SOGP,
                                                               levels_per_run=levels_per_run,
                                                               gp_mu=20, gp_lambda=20,
                                                               gp_crossover_probability=0.5,
                                                               gp_mutation_probability=0.5,
                                                               gp_generations=50, es_generations=100,
                                                               maximum_block_size=maximum_block_size,
                                                               required_convergence=required_convergence,
                                                               restart_from_checkpoint=restart_from_checkpoint)
    program_generator.initialize_code_generation(max_level)
    program_generator.generate_l3_file(program)
    program_generator.run_exastencils_compiler()
    program_generator.run_c_compiler()
    runtime, convergence_factor = program_generator.evaluate(infinity, 10)
    print(f'Runtime: {runtime}, Convergence factor: {convergence_factor}\n', flush=True)
    print(f'ExaSlang representation:\n{program}\n', flush=True)
    log_dir_name = f'{problem_name}/data'
    if not os.path.exists(log_dir_name):
        os.makedirs(log_dir_name)
    i = 1
    for log in stats:
        pickle.dump(log, open(f"{log_dir_name}/log{i}.p", "wb"))
        i += 1
    program_generator.restore_files()


if __name__ == "__main__":
    main()

