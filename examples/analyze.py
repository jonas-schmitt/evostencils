from evostencils.optimization.program import Optimizer
from evostencils.evaluation.performance import PerformanceEvaluator
from evostencils.code_generation.exastencils import ProgramGenerator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from deap import tools
from matplotlib import rc
import matplotlib.font_manager as font_manager


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
    epsilon = 1e-7
    problem_name = program_generator.problem_name
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators,
                          fields,
                          performance_evaluator=performance_evaluator, program_generator=program_generator,
                          epsilon=epsilon, infinity=infinity)
    path = "../helmholtz-results/results"
    # path = "../helmholtz-results/results_k1"
    nexperiments = 10
    # plot_average_solving_time(optimizer, path, "average-solving-time.pdf", nexperiments)
    # plot_minimum_solving_time(optimizer, path, "minimum-solving-time.pdf", nexperiments)
    plot_pareto_front(optimizer, path, nexperiments)

    v01 = [(160.0, 6378.263999999999, "V(0, 1)"), (320.0, 35110.030000000006, "V(0, 1)")]
    f01 = [(160.0, 8146.4039999999995, "F(0, 1)"), (320.0, 42870.44, "F(0, 1)")]
    v11 = [(160.0, 7664.483, "V(1, 1)"), (320.0, 44274.6, "V(1, 1)")]
    evo5 = [(160.0, 5012.040999999999, "EP-5"), (320.0, 28388.24, "EP-5"), (640.0, 227730.8, "EP-5")]
    evo2 = [(160.0, 7862.6630000000005, "EP-2"), (320.0, 29893.72, "EP-2"), (640.0, 241726.4, "EP-2")]
    evo10 = [(160.0, 6696.887999999999, "EP-10"), (320.0, 31385.890000000003, "EP-10"), (640.0, 246071.2, "EP-10")]
    columns = ['k', r'Solving Time per Grid Point (\textmu s)', 'Preconditioner']
    data = []
    data += evo2
    data += evo5
    data += evo10
    data += v01
    data += v11
    data += f01
    data = [(x[0], x[1] / (2**7 * x[0]/80 - 1)**2*1e3 , x[2]) for x in data]

    df = pd.DataFrame(data, columns=columns)
    rc('text', usetex=True)
    # rc('font', **{'family': 'serif', 'serif': ['Times']})
    sns.set_context("paper")
    sns.set_style('whitegrid', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    palette = sns.color_palette("colorblind", n_colors=6)
    tmp = sns.relplot(x=columns[0], y=columns[1], hue=columns[2], style=columns[2],
                data=df, kind='line', dashes=False, markers=["s"]*6, palette=palette, legend=False)

    tmp.ax.legend(labels=["EP-2", "EP-5", "EP-10", "V(0, 1)", "V(1, 1)", "F(0, 1)"], loc=2)
    # plt.ylim(0, 10000)

    # sns.relplot(x=columns[0], y=columns[1], hue=columns[3], style=columns[2],
    #             data=df, kind='line', markers=['o', 'X', 's'], dashes=False, palette=palette)
    # plt.xlim(0, 35)
    plt.tight_layout()
    plt.savefig("solving-time.pdf", quality=100, dpi=300)


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
    columns = ['Number of Iterations', 'Execution Time per Iteration (ms)', 'Experiment']
    df1 = pd.DataFrame(front, columns=columns)
    df2 = pd.DataFrame(global_front, columns=columns)
    rc('text', usetex=True)
    sns.set_context("paper")
    sns.set_style('ticks', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    # sns.despine()
    fig, ax = plt.subplots()
    # palette = sns.color_palette("colorblind", n_colors=number_of_experiments)
    sns.scatterplot(x=columns[0], y=columns[1], style=columns[2], hue=columns[2],
                    data=df1, markers=['o']*(number_of_experiments), legend=False, ax=ax)
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

