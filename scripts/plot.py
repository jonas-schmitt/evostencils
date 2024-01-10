from evostencils.optimization.program import Optimizer
from evostencils.model_based_prediction.performance import PerformanceEvaluator
from evostencils.code_generation.exastencils import ProgramGenerator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from deap import tools
from matplotlib import rc
from mpi4py import MPI
import matplotlib.font_manager as font_manager
import evostencils

def main():

    # TODO adapt to actual path to exastencils project
    cwd = os.path.dirname(os.path.dirname(evostencils.__file__))
    # Path to the ExaStencils compiler
    compiler_path = f'{cwd}/../exastencils/Compiler/Compiler.jar'
    # Path to base folder
    base_path = f'{cwd}/example_problems'
    # Relative path to platform file (from base folder)
    platform_path = f'lib/linux.platform'
    # Example problem from L2
    # Relative path to settings file (from base folder)
    settings_path = f'Poisson/2D_FD_Poisson_fromL2.settings'
    # Relative path to knowledge file (from base folder)
    knowledge_path = f'Poisson/2D_FD_Poisson_fromL2.knowledge'
    # Name of the multigrid cycle function
    cycle_name = "gen_mgCycle"  # Default name on L2
    # Additional global parameter values within the PDE system
    pde_parameter_values = None
    # The maximum number of iterations considered acceptable for a solver
    solver_iteration_limit = 500
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
    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path, platform_path, mpi_rank,
                                             cycle_name=cycle_name, solver_iteration_limit=solver_iteration_limit)
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

    # lfa_grids = [lfa_lab.Grid(dimension, g.step_size) for g in finest_grid]
    # convergence_evaluator = ConvergenceEvaluator(dimension, coarsening_factors, lfa_grids)
    bytes_per_word = 8
    # Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz
    peak_performance = 26633.33 * 1e6
    peak_bandwidth = 26570.26 * 1e6
    # Measured on the target platform
    runtime_coarse_grid_solver = 2.833324499999999 * 1e-3
    performance_evaluator = PerformanceEvaluator(peak_performance, peak_bandwidth, bytes_per_word,
                                                 runtime_coarse_grid_solver=runtime_coarse_grid_solver)
    infinity = 1e100
    epsilon = 1e-6
    problem_name = program_generator.problem_name
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                          mpi_comm=comm, mpi_rank=mpi_rank, number_of_mpi_processes=nprocs,
                          program_generator=program_generator,
                          convergence_evaluator=convergence_evaluator,
                          performance_evaluator=performance_evaluator,
                          epsilon=epsilon, infinity=infinity, checkpoint_directory_path=checkpoint_directory_path)
    path = f"../gpem-21-results/{problem_name}"
    nexperiments = 10
    plot_pareto_front(optimizer, path, nexperiments)

def plot_pareto_front(optimizer, base_path='./', number_of_experiments=1):
    hofs = []
    front = []
    global_hof = tools.ParetoFront()
    for i in range(number_of_experiments):
        pop = optimizer.load_data_structure(f'{base_path}/data_{i}/pop_0.p')
        hof = tools.ParetoFront()
        hof.update(pop)
        hofs.append(hof)
        front += [(*ind.fitness.values, 1) for ind in hof]
        global_hof.update(pop)

    global_front = [(*ind.fitness.values, 'red') for ind in global_hof]
    columns = ['Convergence Factor', 'Execution Time per Iteration (ms)', 'Experiment']
    df1 = pd.DataFrame(front, columns=columns)
    df2 = pd.DataFrame(global_front, columns=columns)
    rc('text', usetex=True)
    sns.set_context("paper")
    sns.set_style('ticks', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    # sns.despine()
    fig, ax = plt.subplots()
    # palette = sns.color_palette("colorblind", n_colors=number_of_experiments)
    # sns.scatterplot(x=columns[0], y=columns[1], style=columns[2], hue=columns[2],
    #                 data=df1, markers=['o']*(number_of_experiments), legend=False, ax=ax)
    sns.scatterplot(x=columns[0], y=columns[1], style=columns[2], hue=columns[2],
                    data=df1, legend=False, ax=ax)
    sns.lineplot(x=columns[0], y=columns[1], color='red',
                data=df2, legend=False,
                ax=ax)
    # plt.xlim(3000, 10000)
    # plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("pareto-front.pdf", quality=100, dpi=300)

def plot_minimum_number_of_iterations(optimizer, base_path='./', number_of_experiments=1):
    data = []
    for i in range(number_of_experiments):
        log = optimizer.load_data_structure(f'{base_path}/data_{i}/log_0.p')
        minimum_iterations = log.chapters["number_of_iterations"].select("min")
        stats = [(gen+1, iterations, i+1) for gen, iterations in enumerate(minimum_iterations) if iterations < 10000]
        data += stats
    columns = ['Generation', 'Minimum Number of Iterations', 'Experiment']
    df = pd.DataFrame(data, columns=columns)
    rc('text', usetex=True)
    plt.rcParams["font.family"] = "Times New Roman"
    sns.set(style="white")
    sns.set_context("paper")
    sns.despine()
    palette = sns.color_palette("colorblind", n_colors=number_of_experiments)
    sns.relplot(x=columns[0], y=columns[1], hue=columns[2],
                data=df, kind='line', dashes=False, palette=palette, legend=False)
    # plt.ylim(0, 10000)

    # sns.relplot(x=columns[0], y=columns[1], hue=columns[3], style=columns[2],
    #             data=df, kind='line', markers=['o', 'X', 's'], dashes=False, palette=palette)
    # plt.xlim(0, 35)
    plt.tight_layout()
    plt.savefig("minimum-iterations.pdf", quality=100, dpi=300)

def plot_minimum_runtime_per_iteration(optimizer, base_path='./', number_of_experiments=1):
    data = []
    for i in range(number_of_experiments):
        log = optimizer.load_data_structure(f'{base_path}/data_{i}/log_0.p')
        minimum_runtime = log.chapters["execution_time"].select("min")
        stats = [(gen+1, runtime, i+1) for gen, runtime in enumerate(minimum_runtime) if runtime < 10000]
        data += stats
    columns = ['Generation', 'Minimum Execution Time per Iteration (ms)', 'Experiment']
    df = pd.DataFrame(data, columns=columns)
    rc('text', usetex=True)
    plt.rcParams["font.family"] = "Times New Roman"
    sns.set(style="white")
    sns.set_context("paper")
    sns.despine()
    # palette = sns.color_palette("colorblind", n_colors=3)
    palette = sns.color_palette("colorblind", n_colors=number_of_experiments)
    sns.relplot(x=columns[0], y=columns[1], hue=columns[2],
                data=df, kind='line', dashes=False, palette=palette, legend=False)
    # plt.ylim(0, 10000)

    # sns.relplot(x=columns[0], y=columns[1], hue=columns[3], style=columns[2],
    #             data=df, kind='line', markers=['o', 'X', 's'], dashes=False, palette=palette)
    # plt.xlim(0, 35)
    plt.tight_layout()
    plt.savefig("minimum-runtime-per-iteration.pdf", quality=100, dpi=300)

def plot_minimum_solving_time(optimizer, base_path='./', file_name=None, number_of_experiments=10, min_generation=0, max_generation=150):
    data = []
    for i in range(number_of_experiments):
        log = optimizer.load_data_structure(f'{base_path}/data_{i}/log_0.p')
        runtime = log.chapters["execution_time"].select("min")
        runtime = runtime[min_generation:max_generation]
        number_of_iterations = log.chapters["number_of_iterations"].select("min")
        number_of_iterations = number_of_iterations[min_generation:max_generation]
        stats = [(gen+1, tmp[0]*tmp[1]*1e-3, i+1) for gen, tmp in enumerate(zip(number_of_iterations, runtime)) if tmp[0]*tmp[1]*1e-3 < 50]
        data += stats
    columns = ['Generation', 'Minimum Solving Time (s)', 'Experiment']
    df = pd.DataFrame(data, columns=columns)
    rc('text', usetex=True)
    plt.rcParams["font.family"] = "Times New Roman"
    sns.set(style="white")
    sns.set_context("paper")
    sns.despine()
    # palette = sns.color_palette("colorblind", n_colors=3)
    palette = sns.color_palette("colorblind", n_colors=number_of_experiments)
    sns.relplot(x=columns[0], y=columns[1], hue=columns[2],
                data=df, kind='line', dashes=False, palette=palette, legend=False)
    plt.ylim(0, 50)

    # sns.relplot(x=columns[0], y=columns[1], hue=columns[3], style=columns[2],
    #             data=df, kind='line', markers=['o', 'X', 's'], dashes=False, palette=palette)
    plt.xlim(0, 150)
    plt.tight_layout()
    if file_name is None:
        file_name = "minimum-solving-time.pdf"
    plt.savefig(file_name, quality=100, dpi=300)


def plot_average_solving_time(optimizer, base_path='./', file_name=None, number_of_experiments=10, min_generation=0, max_generation=150):
    data = []
    for i in range(number_of_experiments):
        log = optimizer.load_data_structure(f'{base_path}/data_{i}/log_0.p')
        runtime = log.chapters["execution_time"].select("avg")
        runtime = runtime[min_generation:max_generation]
        number_of_iterations = log.chapters["number_of_iterations"].select("avg")
        number_of_iterations = number_of_iterations[min_generation:max_generation]
        stats = [(gen+1, tmp[0]*tmp[1]*1e-3, i+1) for gen, tmp in enumerate(zip(number_of_iterations, runtime)) if tmp[0] < 10000]
        data += stats
    columns = ['Generation', 'Average Solving Time (s)', 'Experiment']
    df = pd.DataFrame(data, columns=columns)
    rc('text', usetex=True)
    plt.rcParams["font.family"] = "Times New Roman"
    sns.set(style="white")
    sns.set_context("paper")
    sns.despine()
    # palette = sns.color_palette("colorblind", n_colors=3)
    palette = sns.color_palette("colorblind", n_colors=number_of_experiments)
    sns.relplot(x=columns[0], y=columns[1], hue=columns[2],
                data=df, kind='line', dashes=False, palette=palette, legend=False)
    # plt.ylim(0, 10000)

    # sns.relplot(x=columns[0], y=columns[1], hue=columns[3], style=columns[2],
    #             data=df, kind='line', markers=['o', 'X', 's'], dashes=False, palette=palette)
    plt.xlim(0, 150)
    plt.tight_layout()
    if file_name is None:
        file_name = "average-solving-time.pdf"
    plt.savefig(file_name, quality=100, dpi=300)


if __name__ == "__main__":
    main()

