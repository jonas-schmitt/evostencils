from pytest import *
from evostencils.optimization.program import Optimizer
from evostencils.code_generation.exastencils import ProgramGenerator
import os
import sys
import evostencils
from mpi4py import MPI

def test_optimize_poisson():
    cwd = os.path.dirname(os.path.dirname(evostencils.__file__))
    compiler_path = f'{cwd}/exastencils/Compiler/Compiler.jar'
    base_path = f'{cwd}/example_problems'
    platform_path = f'lib/linux.platform'
    settings_path = f'Poisson/2D_FD_Poisson_fromL2.settings'
    knowledge_path = f'Poisson/2D_FD_Poisson_fromL2.knowledge'
    cycle_name = "gen_mgCycle"
    pde_parameter_values = None
    solver_iteration_limit = 500
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    mpi_rank = comm.Get_rank()
    tmp = "process"
    model_based_estimation = True
    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path, platform_path, mpi_rank,
                                         cycle_name=cycle_name, model_based_estimation=model_based_estimation,
                                         solver_iteration_limit=solver_iteration_limit)
    dimension = program_generator.dimension
    finest_grid = program_generator.finest_grid
    coarsening_factors = program_generator.coarsening_factor
    min_level = program_generator.min_level
    max_level = program_generator.max_level
    equations = program_generator.equations
    operators = program_generator.operators
    fields = program_generator.fields
    problem_name = program_generator.problem_name
    convergence_evaluator = None
    performance_evaluator = None
    if model_based_estimation:
        from evostencils.model_based_prediction.convergence import ConvergenceEvaluator
        from evostencils.model_based_prediction.performance import PerformanceEvaluator
        convergence_evaluator = ConvergenceEvaluator(dimension, coarsening_factors, finest_grid)
        peak_flops = 16 * 6 * 2.6 * 1e9
        peak_bandwidth = 45.8 * 1e9
        bytes_per_word = 8
        performance_evaluator = PerformanceEvaluator(peak_flops, peak_bandwidth, bytes_per_word)
    if mpi_rank == 0 and not os.path.exists(f'{cwd}/{problem_name}'):
        os.makedirs(f'{cwd}/{problem_name}')
    checkpoint_directory_path = f'{cwd}/{problem_name}/checkpoints_{mpi_rank}'
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                          mpi_comm=comm, mpi_rank=mpi_rank, number_of_mpi_processes=nprocs,
                          program_generator=program_generator,
                          convergence_evaluator=convergence_evaluator,
                          performance_evaluator=performance_evaluator,
                          checkpoint_directory_path=checkpoint_directory_path)
    levels_per_run = max_level - min_level
    if model_based_estimation:
        levels_per_run = 1
    assert levels_per_run <= 5, "Can not optimize more than 5 levels"
    optimization_method = optimizer.NSGAII
    use_random_search = False
    mu_ = 2
    lambda_ = 2
    generations = 5
    population_initialization_factor = 2
    generalization_interval = 50
    crossover_probability = 0.7
    mutation_probability = 1.0 - crossover_probability
    node_replacement_probability = 0.1
    evaluation_samples = 1
    maximum_local_system_size = 4
    continue_from_checkpoint = False
    program, dsl_code, pops, stats, hofs = optimizer.evolutionary_optimization(optimization_method=optimization_method,
                                                                     use_random_search=use_random_search,
                                                                     mu_=mu_, lambda_=lambda_,
                                                                     population_initialization_factor=population_initialization_factor,
                                                                     generations=generations,
                                                                     generalization_interval=generalization_interval,
                                                                     crossover_probability=crossover_probability,
                                                                     mutation_probability=mutation_probability,
                                                                     node_replacement_probability=node_replacement_probability,
                                                                     levels_per_run=levels_per_run,
                                                                     evaluation_samples=evaluation_samples,
                                                                     maximum_local_system_size=maximum_local_system_size,
                                                                     model_based_estimation=model_based_estimation,
                                                                     pde_parameter_values=pde_parameter_values,
                                                                    continue_from_checkpoint=continue_from_checkpoint)
    log = optimizer.load_data_structure('./data_0/log_0.p')
    minimum_runtime = log.chapters["execution_time"].select("min")
    assert minimum_runtime < 100
