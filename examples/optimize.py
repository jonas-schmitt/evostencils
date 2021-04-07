from evostencils.optimization.program import Optimizer
# from evostencils.evaluation.convergence import ConvergenceEvaluator
from evostencils.evaluation.performance import PerformanceEvaluator
from evostencils.code_generation.exastencils import ProgramGenerator
import os
# import lfa_lab
import sys
# import dill
from mpi4py import MPI
# MPI.pickle.__init__(dill.dumps, dill.loads)


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
    # settings_path = f'Stokes/2D_FD_Stokes_fromL2.settings'
    # knowledge_path = f'Stokes/2D_FD_Stokes_fromL2.knowledge'

    # 2D Finite difference discretized linear elasticity
    # settings_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.settings'
    # knowledge_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.knowledge'

    # settings_path = f'Helmholtz/2D_FD_Helmholtz_fromL2.settings'
    # knowledge_path = f'Helmholtz/2D_FD_Helmholtz_fromL2.knowledge'

    # settings_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.settings'
    # knowledge_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.knowledge'
    # cycle_name = "mgCycle"

    cycle_name= "gen_mgCycle"

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    mpi_rank = comm.Get_rank()
    if nprocs > 1:
        tmp = "processes"
    else:
        tmp = "process"
    if mpi_rank == 0:
        print(f"Running {nprocs} MPI {tmp}")

    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path, mpi_rank,
                                         cycle_name=cycle_name, solver_iteration_limit=64)

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

    if mpi_rank == 0 and not os.path.exists(f'{cwd}/{problem_name}'):
        os.makedirs(f'{cwd}/{problem_name}')
    checkpoint_directory_path = f'{cwd}/{problem_name}/checkpoints_{mpi_rank}'
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                          mpi_comm=comm, mpi_rank=mpi_rank, number_of_mpi_processes=nprocs,
                          program_generator=program_generator,
                          epsilon=epsilon, infinity=infinity, checkpoint_directory_path=checkpoint_directory_path)

    # restart_from_checkpoint = True
    restart_from_checkpoint = False
    levels_per_run = max_level - min_level
    assert levels_per_run <= 5, "Can not optimize more than 5 levels"
    required_convergence = 0.5
    maximum_block_size = 8
    optimization_method = optimizer.NSGAII
    if len(sys.argv) > 1:
        if sys.argv[1].upper() == "NSGAII":
            optimization_method = optimizer.NSGAII
        elif sys.argv[1].upper() == "NSGAIII":
            optimization_method = optimizer.NSGAIII
        elif sys.argv[1].upper() == "SOGP":
            optimization_method = optimizer.SOGP
        elif sys.argv[1].upper() == "RANDOM":
            optimization_method = optimizer.multi_objective_random_search

    crossover_probability = 2.0/3.0
    mutation_probability = 1.0 - crossover_probability
    parameter_values = {}
    # Optional: If needed supply additional parameters to the optimization
    # values = [80.0 * 2.0**i for i in range(100)]
    # parameter_values = {'k' : values}
    program, pops, stats, hofs = optimizer.evolutionary_optimization(optimization_method=optimization_method,
                                                                     levels_per_run=levels_per_run,
                                                                     gp_mu=256, gp_lambda=4,
                                                                     gp_crossover_probability=crossover_probability,
                                                                     gp_mutation_probability=mutation_probability,
                                                                     gp_generations=250,
                                                                     maximum_block_size=maximum_block_size,
                                                                     parameter_values=parameter_values,
                                                                     required_convergence=required_convergence,
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
            hof_dir =  f'{log_dir_name}/hof_{i}'
            os.makedirs(hof_dir)
            for j, ind in enumerate(hof):
                with open(f'{hof_dir}/individual_{j}.txt', 'w') as grammar_file:
                    grammar_file.write(str(ind) + '\n')


if __name__ == "__main__":
    main()

