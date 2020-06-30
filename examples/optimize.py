from evostencils.optimization.program import Optimizer
from evostencils.evaluation.convergence import ConvergenceEvaluator
from evostencils.evaluation.performance import PerformanceEvaluator
from evostencils.code_generation.exastencils import ProgramGenerator
import os
import lfa_lab
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
    # cycle_name = "VCycle"

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
                                         cycle_name=cycle_name)

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
    # Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz
    peak_performance = 26633.33 * 1e6
    peak_bandwidth = 26570.26 * 1e6
    # Measured on the target platform
    runtime_coarse_grid_solver = 2.833324499999999 * 1e-3
    performance_evaluator = PerformanceEvaluator(peak_performance, peak_bandwidth, bytes_per_word,
                                                 runtime_coarse_grid_solver=runtime_coarse_grid_solver)
    infinity = 1e100
    epsilon = 1e-12
    problem_name = program_generator.problem_name

    if not os.path.exists(f'{cwd}/{problem_name}'):
        os.makedirs(f'{cwd}/{problem_name}')
    checkpoint_directory_path = f'{cwd}/{problem_name}/checkpoints_{mpi_rank}'
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                          mpi_comm=comm, mpi_rank=mpi_rank, number_of_mpi_processes=nprocs,
                          convergence_evaluator=convergence_evaluator,
                          performance_evaluator=performance_evaluator, program_generator=program_generator,
                          epsilon=epsilon, infinity=infinity, checkpoint_directory_path=checkpoint_directory_path)

    # restart_from_checkpoint = True
    restart_from_checkpoint = False
    levels_per_run = max_level - min_level
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
    minimum_solver_iterations = 2**3
    maximum_solver_iterations = 2**10
    # krylov_subspace_methods = ('ConjugateGradient', 'BiCGStab', 'MinRes', 'ConjugateResidual')
    krylov_subspace_methods = ()
    program, pops, stats = optimizer.evolutionary_optimization(optimization_method=optimization_method,
                                                               levels_per_run=levels_per_run,
                                                               gp_mu=100, gp_lambda=100,
                                                               gp_crossover_probability=crossover_probability,
                                                               gp_mutation_probability=mutation_probability,
                                                               gp_generations=100, es_generations=150,
                                                               maximum_block_size=maximum_block_size,
                                                               required_convergence=required_convergence,
                                                               restart_from_checkpoint=restart_from_checkpoint,
                                                               krylov_subspace_methods=krylov_subspace_methods,
                                                               minimum_solver_iterations=minimum_solver_iterations,
                                                               maximum_solver_iterations=maximum_solver_iterations)
    
    if mpi_rank == 0:
        print(f'ExaSlang representation:\n{program}\n', flush=True)
    log_dir_name = f'{problem_name}/data_{mpi_rank}'
    if not os.path.exists(log_dir_name):
        os.makedirs(log_dir_name)
    for i, log in enumerate(stats):
        optimizer.dump_data_structure(log, f"{log_dir_name}/log_{i}.p")
    for i, pop in enumerate(pops):
        optimizer.dump_data_structure(pop, f"{log_dir_name}/pop_{i}.p")


if __name__ == "__main__":
    main()

