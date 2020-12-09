from evostencils.optimization.program import Optimizer
from evostencils.evaluation.performance import PerformanceEvaluator
from evostencils.code_generation.exastencils import ProgramGenerator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from deap import tools
from matplotlib import rc


def main():

    # TODO adapt to actual path to exastencils project

    cwd = os.getcwd()
    compiler_path = f'{cwd}/../exastencils/Compiler/Compiler.jar'
    base_path = f'{cwd}/../exastencils/Examples'

    # 2D Finite difference discretized Poisson
    # settings_path = f'Poisson/2D_FD_Poisson_fromL2.settings'
    # knowledge_path = f'Poisson/2D_FD_Poisson_fromL2.knowledge'

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

    settings_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.settings'
    knowledge_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.knowledge'
    cycle_name = "VCycle"

    settings_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.settings'
    knowledge_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.knowledge'
    cycle_name = "VCycle"

    # cycle_name= "gen_mgCycle"

    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path, cycle_name=cycle_name)

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
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators,
                          fields,
                          performance_evaluator=performance_evaluator, program_generator=program_generator,
                          epsilon=epsilon, infinity=infinity)
    path = "../helmholtz-results/results_05"
    nexperiments = 5
    plot_average_number_of_iterations(optimizer, path, nexperiments)
    plot_average_runtime_per_iteration(optimizer, path, nexperiments)
    plot_average_execution_time(optimizer, path, nexperiments)
    plot_pareto_front(optimizer, path, nexperiments)


def plot_pareto_front(optimizer, base_path='./', number_of_experiments=1):
    hofs = []
    front = []
    for i in range(number_of_experiments):
        pop = optimizer.load_data_structure(f'{base_path}/data_{i}/pop_0.p')
        hof = tools.ParetoFront(similar=lambda a, b: str(a) == str(b))
        hof.update(pop)
        hofs.append(hof)
        front += [(*ind.fitness.values, i + 1) for ind in hof]

    columns = ['Number of Iterations', 'Runtime per Iteration (ms)', 'Experiment']
    df = pd.DataFrame(front, columns=columns)
    rc('text', usetex=True)
    sns.set(style="white")
    sns.set_context("paper")
    sns.despine()
    # palette = sns.color_palette("colorblind", n_colors=3)
    palette = sns.color_palette("colorblind", n_colors=number_of_experiments)
    sns.relplot(x=columns[0], y=columns[1], hue=columns[2], style=columns[2],
                data=df, kind='line', markers=True, dashes=False, palette=palette, legend=False)

    # sns.relplot(x=columns[0], y=columns[1], hue=columns[3], style=columns[2],
    #             data=df, kind='line', markers=['o', 'X', 's'], dashes=False, palette=palette)
    # plt.xlim(0, 35)
    plt.tight_layout()
    plt.savefig("pareto-front.pdf", quality=100, dpi=300)

def plot_average_number_of_iterations(optimizer, base_path='./', number_of_experiments=1):
    data = []
    for i in range(number_of_experiments):
        log = optimizer.load_data_structure(f'{base_path}/data_{i}/log_0.p')
        minimum_iterations = log.chapters["number_of_iterations"].select("avg")
        stats = [(gen+1, iterations, i+1) for gen, iterations in enumerate(minimum_iterations) if iterations < 10000]
        data += stats
    columns = ['Generation', 'Average Number of Iterations', 'Experiment']
    df = pd.DataFrame(data, columns=columns)
    rc('text', usetex=True)
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
    plt.savefig("average-iterations.pdf", quality=100, dpi=300)

def plot_average_runtime_per_iteration(optimizer, base_path='./', number_of_experiments=1):
    data = []
    for i in range(number_of_experiments):
        log = optimizer.load_data_structure(f'{base_path}/data_{i}/log_0.p')
        minimum_runtime = log.chapters["execution_time"].select("avg")
        stats = [(gen+1, runtime, i+1) for gen, runtime in enumerate(minimum_runtime) if runtime < 10000]
        data += stats
    columns = ['Generation', 'Average Runtime per Iteration (ms)', 'Experiment']
    df = pd.DataFrame(data, columns=columns)
    rc('text', usetex=True)
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
    plt.savefig("average-runtime-per-iteration.pdf", quality=100, dpi=300)

def plot_average_execution_time(optimizer, base_path='./', number_of_experiments=1):
    data = []
    for i in range(number_of_experiments):
        log = optimizer.load_data_structure(f'{base_path}/data_{i}/log_0.p')
        minimum_runtime = log.chapters["execution_time"].select("avg")
        minimum_number_of_iterations = log.chapters["number_of_iterations"].select("avg")
        stats = [(gen+1, tmp[0]*tmp[1], i+1) for gen, tmp in enumerate(zip(minimum_number_of_iterations, minimum_runtime)) if tmp[0] < 10000]
        data += stats
    columns = ['Generation', 'Average Execution Time (ms)', 'Experiment']
    df = pd.DataFrame(data, columns=columns)
    rc('text', usetex=True)
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
    plt.savefig("average-execution-time.pdf", quality=100, dpi=300)


if __name__ == "__main__":
    main()

