from evostencils.optimization.program import Optimizer
from evostencils.code_generation.hyteg import ProgramGenerator
import evostencils.grammar.multigrid as mg_grammar
import os
import sys
from mpi4py import MPI


def main():
    cwd = f'{os.getcwd()}'
    eval_software = "hyteg"

    # I. Set up MPI
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    mpi_rank = comm.Get_rank()
    if nprocs > 1:
        tmp = "processes"
    else:
        tmp = "process"
    if mpi_rank == 0:
        print(f"Running {nprocs} MPI {tmp}")

    # II. problem specifications
    problem_name = "2dpoisson"
    flexmg_min_level = 0
    flexmg_max_level = 4
    cgs_level = 0
    mg_grammar.optimize_cgs = False # optimises the tolerance and level of the coarse-grid solver
    if mg_grammar.optimize_cgs:
        mg_grammar.cgs_tolerance = [1e-3, 1e-5, 1e-7, 1e-9] # set the options for the tolerance
        mg_grammar.cgs_level = [0, 1, 2] # set the options for the level
    mg_grammar.optimize_cgc_scalingfactor = False # optimises the weights / scaling factors for coarse-grid correction (set to False with hyteg)
    assert flexmg_min_level < flexmg_max_level
    assert flexmg_min_level >= cgs_level
    assert flexmg_max_level - flexmg_min_level < 5
    if eval_software == "hyteg":
        assert not mg_grammar.optimize_cgc_scalingfactor
    program_generator = ProgramGenerator(flexmg_min_level,flexmg_max_level , mpi_rank, cgs_level)


    if mpi_rank == 0 and not os.path.exists(f'{cwd}/{problem_name}'):
        # Create directory for checkpoints and output data
        os.makedirs(f'{cwd}/{problem_name}')
    # Path to directory for storing checkpoints
    checkpoint_directory_path = f'{cwd}/{problem_name}/checkpoints_{mpi_rank}'


    # III. Create optimizer object
    optimizer = Optimizer(flexmg_min_level, flexmg_max_level, mpi_comm=comm, mpi_rank=mpi_rank, number_of_mpi_processes=nprocs,
                          program_generator=program_generator, checkpoint_directory_path=checkpoint_directory_path)

    # IV. optimization parameters
    optimization_method = optimizer.NSGAII
    mu_ = 4 # Population size
    lambda_ = 4 # Number of offspring
    generations = 4  # Number of generations
    population_initialization_factor = 1  # Multiply mu_ by this factor to set the initial population size
    generalization_interval = 150
    crossover_probability = 0.7
    mutation_probability = 1.0 - crossover_probability
    node_replacement_probability = 0.1  # Probability to perform mutation by altering a single node in the tree
    evaluation_samples = 1  # Number of evaluation samples

    # V. Return values of the optimization
    # program: program string representing the multigrid solver functions
    # pops: Populations at the end of each optimization run on the respective subrange of the discretization hierarchy
    # stats: Statistics structure (data structure provided by the DEAP framework)
    # hofs: Hall-of-fames at the end of each optimization run on the respective subrange of the discretization hierarchy
    program, dsl_code, pops, stats, hofs, fitnesses = optimizer.evolutionary_optimization(optimization_method=optimization_method,
                                                                     use_random_search=False,
                                                                     mu_=mu_, lambda_=lambda_,
                                                                     population_initialization_factor=population_initialization_factor,
                                                                     generations=generations,
                                                                     generalization_interval=generalization_interval,
                                                                     crossover_probability=crossover_probability,
                                                                     mutation_probability=mutation_probability,
                                                                     node_replacement_probability=node_replacement_probability,
                                                                     levels_per_run=flexmg_max_level - flexmg_min_level,
                                                                     evaluation_samples=evaluation_samples,
                                                                     maximum_local_system_size=1,
                                                                     continue_from_checkpoint=False)
    # VI. Print the outcome of the optimization and store the data and statistics
    if mpi_rank == 0:
        print(f'\n{eval_software} specifications:\n{dsl_code}\n', flush=True)
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
        optimizer.dump_data_structure(fitnesses, f"{log_dir_name}/fitnesses.p")


if __name__ == "__main__":
    main()
