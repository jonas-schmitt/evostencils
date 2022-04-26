from evostencils.optimization.program import Optimizer
from evostencils.code_generation.exastencils_FAS import ProgramGeneratorFAS
from evostencils.code_generation.exastencils import ProgramGenerator
import os
import sys
from mpi4py import MPI



def main():
    cwd = f'{os.getcwd()}'
    # Path to the ExaStencils compiler
    compiler_path = f'{cwd}/../exastencils/Compiler/Compiler.jar'
    # Path to base folder
    base_path = f'{cwd}/example_problems'
    # Relative path to platform file (from base folder)
    platform_path = f'lib/linux.platform'
    # Example problem from L2
    # Relative path to settings file (from base folder)
    # settings_path = f'Poisson/2D_FD_Poisson_fromL2.settings'
    # settings_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.settings'
    # settings_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.settings'
    settings_path = f'FAS_2D_Basic/FAS_2D_Basic.settings'
    # Relative path to knowledge file (from base folder)
    # knowledge_path = f'Poisson/2D_FD_Poisson_fromL2.knowledge'
    # knowledge_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.knowledge'
    # knowledge_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.knowledge'
    knowledge_path = f'FAS_2D_Basic/FAS_2D_Basic.knowledge'
    # Name of the multigrid cycle function
    cycle_name = "gen_mgCycle"  # Default name
    # Additional global parameter values within the PDE system
    pde_parameter_values = None
    # The maximum number of iterations considered acceptable for a solver
    solver_iteration_limit = 500
    FAS = False
    # Hacky solution for now
    if "FAS" in knowledge_path or "FAS" in settings_path:
        FAS = True

    # Example problem from L3
    # Warning: Currently not working, due to a bug in the ExaStencils compiler!
    # settings_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.settings'
    # knowledge_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.knowledge'
    # cycle_name = "mgCycle"
    # values = [80.0 * 2.0**i for i in range(100)]
    # pde_parameter_values = {'k': values}
    # solver_iteration_limit = 10000

    # Set up MPI
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
    # Create program generator object
    if not FAS:
        program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path, platform_path, mpi_rank,
                                             cycle_name=cycle_name, use_jacobi_prefix=use_jacobi_prefix,
                                             solver_iteration_limit=solver_iteration_limit)
    else:
        # Warning: FAS Support experimental, requires adaption of the symbol names provided to this class
        program_generator = ProgramGeneratorFAS('FAS_2D_Basic', 'Solution', 'RHS', 'Residual', 'Approximation',
                                                'RestrictionNode', 'CorrectionNode',
                                                'Laplace', 'gamSten', cycle_name, 'CGS', 'Smoother', mpi_rank=mpi_rank)

    # Obtain extracted information from program generator
    dimension = program_generator.dimension  # Dimensionality of the problem
    finest_grid = program_generator.finest_grid  # Representation of the finest grid
    coarsening_factors = program_generator.coarsening_factor
    min_level = program_generator.min_level  # Minimum discretization level
    max_level = program_generator.max_level  # Maximum discretization level
    equations = program_generator.equations  # System of PDEs in SymPy
    operators = program_generator.operators  # Discretized differential operators
    fields = program_generator.fields  # Variables that occur within system of PDEs
    infinity = 1e100  # Upper limit that is considered infinite
    epsilon = 1e-12  # Lower limit that is considered zero
    problem_name = program_generator.problem_name
    convergence_evaluator = None
    performance_evaluator = None
    if model_based_estimation:
        # Create convergence and performance evaluator objects
        # Only needed when a model-based estimation should be used within the optimization
        # (Not recommended due to the limitations, but useful for testing)
        from evostencils.evaluation.convergence import ConvergenceEvaluator
        from evostencils.evaluation.performance import PerformanceEvaluator
        convergence_evaluator = ConvergenceEvaluator(dimension, coarsening_factors, finest_grid)
        # Peak FLOP performance of the machine
        peak_flops = 16 * 6 * 2.6 * 1e9
        # Peak memory bandwidth of the machine
        peak_bandwidth = 45.8 * 1e9
        # Number of bytes per word
        bytes_per_word = 8  # Double = 64 Bit = 8 Bytes
        performance_evaluator = PerformanceEvaluator(peak_flops, peak_bandwidth, bytes_per_word)
    if mpi_rank == 0 and not os.path.exists(f'{cwd}/{problem_name}'):
        # Create directory for checkpoints and output data
        os.makedirs(f'{cwd}/{problem_name}')
    # Path to directory for storing checkpoints
    checkpoint_directory_path = f'{cwd}/{problem_name}/checkpoints_{mpi_rank}'
    # Create optimizer object
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                          mpi_comm=comm, mpi_rank=mpi_rank, number_of_mpi_processes=nprocs,
                          program_generator=program_generator,
                          convergence_evaluator=convergence_evaluator,
                          performance_evaluator=performance_evaluator,
                          epsilon=epsilon, infinity=infinity, checkpoint_directory_path=checkpoint_directory_path)
    # Option to split the optimization into multiple runs,
    # where each run is only performed on a subrange of the discretization hierarchy starting at the top (finest grid)
    # (Not recommended for code-generation based evaluation)
    levels_per_run = max_level - min_level
    if model_based_estimation:
        # Model-based estimation only feasible for up to 2 levels per run
        levels_per_run = 1
    assert levels_per_run <= 5, "Can not optimize more than 5 levels"
    # Choose optimization method
    optimization_method = optimizer.NSGAII
    if len(sys.argv) > 1:
        # Multi-objective (mu+lambda)-EA with NSGA-II non-dominated sorting-based selection
        if sys.argv[1].upper() == "NSGAII":
            optimization_method = optimizer.NSGAII
        # Multi-objective (mu+lambda)-EA with NSGA-III non-dominated sorting-based selection
        elif sys.argv[1].upper() == "NSGAIII":
            optimization_method = optimizer.NSGAIII
        # Classic single-objective (mu+lambda)-EA with binary tournament selection
        elif sys.argv[1].upper() == "SOGP":
            optimization_method = optimizer.SOGP
    # Option to use random search instead of crossover and mutation to create new individuals
    use_random_search = False

    mu_ = 32  # Population size
    lambda_ = 32  # Number of offspring
    generations = 50  # Number of generations
    population_initialization_factor = 2  # Multiply mu_ by this factor to set the initial population size

    # Number of generations after which a generalization is performed
    # This is achieved by incrementing min_level and max_level within the optimization
    # Such that a larger (and potentially more difficult) instance of the same problem is considered in subsequent generations
    generalization_interval = 50
    crossover_probability = 0.7
    mutation_probability = 1.0 - crossover_probability
    node_replacement_probability = 0.1  # Probability to perform mutation by altering a single node in the tree
    evaluation_samples = 3  # Number of evaluation samples
    maximum_local_system_size = 4  # Maximum size of the local system solved within each step of a block smoother
    # Option to continue from the checkpoint of a previous optimization
    # Warning: So far no check is performed whether the checkpoint is compatible with the current optimization setting
    continue_from_checkpoint = False

    # Return values of the optimization
    # program: ExaSlang program string representing the multigrid solver functions
    # pops: Populations at the end of each optimization run on the respective subrange of the discretization hierarchy
    # stats: Statistics structure (data structure provided by the DEAP framework)
    # hofs: Hall-of-fames at the end of each optimization run on the respective subrange of the discretization hierarchy
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
                                                                     maximum_local_system_size=maximum_local_system_size,
                                                                     model_based_estimation=model_based_estimation,
                                                                     pde_parameter_values=pde_parameter_values,
                                                                     continue_from_checkpoint=continue_from_checkpoint)
    # Print the outcome of the optimization and store the data and statistics
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
