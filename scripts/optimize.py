from evostencils.optimization.program import Optimizer
from evostencils.code_generation.exastencils import ProgramGenerator
import os
import sys
from mpi4py import MPI

def main():

    # TODO adapt to actual path to exastencils project

    cwd = os.getcwd()
    compiler_path = f'{cwd}/../exastencils/Compiler/Compiler.jar'
    base_path = f'{cwd}/example_problems'
    # Example problem from L2
    settings_path = f'Poisson/2D_FD_Poisson_fromL2.settings'
    knowledge_path = f'Poisson/2D_FD_Poisson_fromL2.knowledge'
    cycle_name = "gen_mgCycle"
    pde_parameter_values = None
    solver_iteration_limit = 500

    # Example problem from L3
    # settings_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.settings'
    # knowledge_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.knowledge'
    # cycle_name = "mgCycle"
    # values = [80.0 * 2.0**i for i in range(100)]
    # pde_parameter_values = {'k': values}
    # solver_iteration_limit = 10000


    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    mpi_rank = comm.Get_rank()
    if nprocs > 1:
        tmp = "processes"
    else:
        tmp = "process"
    if mpi_rank == 0:
        print(f"Running {nprocs} MPI {tmp}")


    model_based_estimation = False
    use_jacobi_prefix = True
    # Experimental and not recommended:
    # Use model based estimation instead of code generation and evaluation
    # model_based_estimation = True
    if model_based_estimation:
        # LFA based estimation inaccurate with jacobi prefix
        use_jacobi_prefix = False
    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path, mpi_rank,
                                         cycle_name=cycle_name, use_jacobi_prefix=use_jacobi_prefix,
                                         solver_iteration_limit=solver_iteration_limit)

    # Obtain extracted information from program generator
    dimension = program_generator.dimension
    finest_grid = program_generator.finest_grid
    coarsening_factors = program_generator.coarsening_factor
    min_level = program_generator.min_level
    max_level = program_generator.max_level
    equations = program_generator.equations
    operators = program_generator.operators
    fields = program_generator.fields
    infinity = 1e100
    epsilon = 1e-12
    problem_name = program_generator.problem_name
    convergence_evaluator = None
    performance_evaluator = None
    if model_based_estimation:
        from evostencils.evaluation.convergence import ConvergenceEvaluator
        from evostencils.evaluation.performance import PerformanceEvaluator
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
                          epsilon=epsilon, infinity=infinity, checkpoint_directory_path=checkpoint_directory_path)

    levels_per_run = max_level - min_level
    if model_based_estimation:
        levels_per_run = 1
    assert levels_per_run <= 5, "Can not optimize more than 5 levels"
    optimization_method = optimizer.NSGAII
    if len(sys.argv) > 1:
        if sys.argv[1].upper() == "NSGAII":
            optimization_method = optimizer.NSGAII
        elif sys.argv[1].upper() == "NSGAIII":
            optimization_method = optimizer.NSGAIII
        elif sys.argv[1].upper() == "SOGP":
            optimization_method = optimizer.SOGP
    use_random_search = False
    mu_ = 16
    lambda_ = 16
    generations = 50
    population_initialization_factor = 1
    generalization_interval = 50
    crossover_probability = 0.7
    mutation_probability = 1.0 - crossover_probability
    node_replacement_probability = 0.1
    evaluation_samples = 3
    maximum_block_size = 4
    restart_from_checkpoint = False
    program, pops, stats, hofs = optimizer.evolutionary_optimization(optimization_method=optimization_method,
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
                                                                     maximum_block_size=maximum_block_size,
                                                                     model_based_estimation=model_based_estimation,
                                                                     pde_parameter_values=pde_parameter_values,
                                                                     restart_from_checkpoint=restart_from_checkpoint)

    if mpi_rank == 0:
        print(f'\nGrammar representation:\n{program}\n', flush=True)
        if not os.path.exists(f'./{problem_name}'):
            os.makedirs(f'./{problem_name}')
        j = 0
        log_dir_name = f'./{problem_name}/data_{j}'
        while os.path.exists(log_dir_name):
            j += 1
            log_dir_name = f'./{problem_name}/data_{j}'
        os.makedirs(log_dir_name)
        for i, log in enumerate(stats):
            optimizer.dump_data_structure(log, f"{log_dir_name}/log_{i}.p")
        for i, pop in enumerate(pops):
            optimizer.dump_data_structure(pop, f"{log_dir_name}/pop_{i}.p")
        for i, hof in enumerate(hofs):
            hof_dir = f'{log_dir_name}/hof_{i}'
            os.makedirs(hof_dir)
            for j, ind in enumerate(hof):
                with open(f'{hof_dir}/individual_{j}.txt', 'w') as grammar_file:
                    grammar_file.write(str(ind) + '\n')


if __name__ == "__main__":
    main()
