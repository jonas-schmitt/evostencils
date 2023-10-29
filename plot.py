from evostencils.optimization.program import Optimizer
from evostencils.code_generation.exastencils import ProgramGenerator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from deap import tools
from matplotlib import rc
import pickle
import math
import numpy as np
import matplotlib.font_manager as font_manager
from mpi4py import MPI
from evostencils.code_generation.exastencils_FAS import ProgramGeneratorFAS
from matplotlib.animation import FuncAnimation

def main():
    # TODO adapt to actual path to exastencils project
    dir_name = 'Poisson'
    problem_name = f'2D_FD_{dir_name}_fromL2'
    cwd = os.getcwd()
    compiler_path = f'{cwd}/../exastencils/Compiler/Compiler.jar'
    base_path = f'{cwd}/./results'

    # Set up MPI
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    mpi_rank = comm.Get_rank()

    # Create program generator object
    program_generator = ProgramGeneratorFAS('FAS_2D_Basic', 'Solution', 'RHS', 'Residual', 'Approximation',
                                            'RestrictionNode', 'CorrectionNode',
                                            'Laplace', 'gamSten', 'mgCycle', 'CGS', None, mpi_rank=mpi_rank)

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
    convergence_evaluator = None
    performance_evaluator = None
    problem_name = program_generator.problem_name
    checkpoint_directory_path = f'{cwd}/{problem_name}/checkpoints_{mpi_rank}'

    # Create optimizer object
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                          mpi_comm=comm, mpi_rank=mpi_rank, number_of_mpi_processes=nprocs,
                          program_generator=program_generator,
                          convergence_evaluator=convergence_evaluator,
                          performance_evaluator=performance_evaluator,
                          epsilon=epsilon, infinity=infinity, checkpoint_directory_path=checkpoint_directory_path)

    plot_fitnesses(optimizer, base_path)


def plot_fitnesses(experiment_folder):
    # unpickle fitnesses
    with open(f'{experiment_folder}/fitnesses.p', 'rb') as file:
        fitness_history = pickle.load(file)
    
    fig, ax = plt.subplots()
    
    def update(generation):
        fitnesses = fitness_history[generation]
        ax.clear()
        # add point for v_cycle
        ax.scatter(v_cycle[0], v_cycle[1], color='red', marker='x')
        # mark v_cycle point with text
        ax.text(v_cycle[0], v_cycle[1], 'V-Cycle', fontsize=8)
        x=[]
        y=[]

        x = [fit[0] for fit in fitnesses]
        y = [fit[1] for fit in fitnesses]
        x_mean = np.mean(np.array(x))
        y_mean = np.mean(np.array(y))
        x_max = np.max(np.array(x))
        y_max = np.max(np.array(y))
        x_min = np.min(np.array(x))
        y_min = np.min(np.array(y))
        # reduce mean, avg and max to 3 decimal places and exponentiate
        x_mean = f'{x_mean:.2e}'
        y_mean = f'{y_mean:.2e}'
        x_max = f'{x_max:.2e}'
        y_max = f'{y_max:.2e}'
        x_min = f'{x_min:.2e}'
        y_min = f'{y_min:.2e}'
       
        
        sns.scatterplot(x=x, y=y, ax=ax)
        # style seaborn plot
        sns.set_context("paper")
        sns.set_style('ticks', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
        sns.despine()

        
        ax.set_xlabel('Convergence Factor')
        ax.set_ylabel('Execution Time Per Iteration (ms)')
        ax.set_title(f'Fitness of Individuals In Population')
        # print generation, mean, min and avg in the top near title, 
        ax.text(0.75, 0.95, f'Mean: ({x_mean},{y_mean})', transform=ax.transAxes,fontsize=8)
        ax.text(0.75, 0.90, f'Min: ({x_min},{y_min})', transform=ax.transAxes,fontsize=8)
        ax.text(0.75, 0.85, f'Max: ({x_max},{y_max})', transform=ax.transAxes,fontsize=8)
        ax.text(0.75, 0.80, f'Generation: {generation}', transform=ax.transAxes,fontsize=8)
      
    # animate
    anim = FuncAnimation(fig, update, frames=range(0, len(fitness_history)))
    anim.show()

def plot_pareto_front(optimizer, base_path='./', number_of_experiments=1):
    def get_total_execution_time(convergence_factor, execution_time_per_iteration):
        infinity = 1e100
        res_reduction_factor = 1e-10
        if convergence_factor < 1:
            return math.log(res_reduction_factor) / math.log(convergence_factor) * execution_time_per_iteration
        else:
            return convergence_factor * math.sqrt(infinity) * execution_time_per_iteration

    def n_iterations(convergence_factor):
        infinity = 1e100
        res_reduction_factor = 1e-10
        if convergence_factor < 1:
            return math.log(res_reduction_factor) / math.log(convergence_factor)
        else:
            return convergence_factor * math.sqrt(infinity)

    def get_execution_time(c, evo_T):
        n = n_iterations(c)
        return evo_T  # (evo_T * 2 ) / (n + 1)

    hofs = []
    front = []
    global_hof = tools.ParetoFront()
    for i in range(1, number_of_experiments + 1):
        pop = optimizer.load_data_structure(f'{base_path}/data_{i}/pop_0.p')
        hof = tools.ParetoFront()
        hof.update(pop)
        hofs.append(hof)
        front += [(ind.fitness.values[0], get_execution_time(ind.fitness.values[0], ind.fitness.values[1]), 1) for ind in hof]
        global_hof.update(pop)

    global_front = [(ind.fitness.values[0], get_execution_time(ind.fitness.values[0], ind.fitness.values[1]), 'red') for ind in global_hof]
    columns = ['Convergence Factor', 'Execution Time per Iteration (ms)', 'Experiment']
    df1 = pd.DataFrame(front, columns=columns)
    df2 = pd.DataFrame(global_front, columns=columns)
    rc('text', usetex=False)
    sns.set_context("paper")
    sns.set_style('ticks', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    # sns.despine()
    fig, ax = plt.subplots()
    # palette = sns.color_palette("colorblind", n_colors=number_of_experiments)
    # sns.scatterplot(x=columns[0], y=columns[1], style=columns[2], hue=columns[2],
    #                 data=df1, markers=['o']*(number_of_experiments), legend=False, ax=ax)
    sns.scatterplot(x=columns[0], y=columns[1], style=columns[2], hue=columns[2],
                    data=df1, legend=False, ax=ax, alpha=0.35)
    sns.lineplot(x=columns[0], y=columns[1], color='red',
                 data=df2, legend=False,
                 ax=ax)
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.1)
    # plt.tight_layout()
    plt.savefig(f"{base_path}/pareto-front-per-iteration.pdf", dpi=300)


def plot_individuals(base_path, number_of_experiments):
    data = []
    i = 11
    with open(f'{base_path}/data_{i}/individuals.p', 'rb') as file:
        log = pickle.load(file)
        for l in log:
            stats = [l.fitness.values]
            data += stats
    columns = ["Convergence Factor", "Minimum Number of Iterations"]
    df = pd.DataFrame(data, columns=columns)
    rc('text', usetex=True)
    sns.set_context("paper")
    sns.set_style('ticks', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    sns.despine()
    sns.scatterplot(x=columns[0], y=columns[1], data=df, legend=False)
    plt.tight_layout()
    plt.savefig(f"{base_path}/individuals.pdf", dpi=300)


def plot_minimum_number_of_iterations(optimizer, base_path='./', number_of_experiments=1):
    data = []
    for i in range(1, number_of_experiments + 1):
        log = optimizer.load_data_structure(f'{base_path}/data_{i}/log_0.p')
        minimum_iterations = log.chapters["number_of_iterations"].select("min")
        stats = [(gen + 1, iterations, i + 1) for gen, iterations in enumerate(minimum_iterations) if iterations < 10000]
        data += stats
    columns = ['Generation', 'Minimum Number of Iterations', 'Experiment']
    df = pd.DataFrame(data, columns=columns)
    rc('text', usetex=False)
    
    plt.rcParams["font.family"] = "Times New Roman"
    sns.set(style="white")
    sns.set_context("paper")
    sns.despine()
    palette = sns.color_palette("colorblind", n_colors=number_of_experiments)
    # sns.relplot(x=columns[0], y=columns[1], hue=columns[2],
    #           data=df, kind='line', dashes=False, palette=palette, legend=False)
    sns.lineplot(x=columns[0], y=columns[1], data=df, legend=False)
    # plt.ylim(0, 10000)

    # sns.relplot(x=columns[0], y=columns[1], hue=columns[3], style=columns[2],
    #             data=df, kind='line', markers=['o', 'X', 's'], dashes=False, palette=palette)
    # plt.xlim(0, 35)
    plt.tight_layout()
    plt.savefig(f"{base_path}/minimum-iterations.pdf", dpi=300)


def plot_minimum_runtime_per_iteration(optimizer, base_path='./', number_of_experiments=1):
    data = []
    for i in range(1, number_of_experiments + 1):
        log = optimizer.load_data_structure(f'{base_path}/data_{i}/log_0.p')
        minimum_runtime = log.chapters["execution_time"].select("min")
        stats = [(gen + 1, runtime, i + 1) for gen, runtime in enumerate(minimum_runtime) if runtime < 10000]
        data += stats
    columns = ['Generation', 'Minimum Execution Time per Iteration (ms)', 'Experiment']
    df = pd.DataFrame(data, columns=columns)
    rc('text', usetex=True)
    sns.set_context("paper")
    sns.set_style('ticks', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    sns.despine()
    # palette = sns.color_palette("colorblind", n_colors=3)
    palette = sns.color_palette("colorblind", n_colors=number_of_experiments)
    # sns.relplot(x=columns[0], y=columns[1], hue=columns[2],
    #           data=df, kind='line', dashes=False, palette=palette, legend=False)
    sns.lineplot(x=columns[0], y=columns[1], data=df, legend=False)
    # plt.ylim(0, 10000)

    # sns.relplot(x=columns[0], y=columns[1], hue=columns[3], style=columns[2],
    #             data=df, kind='line', markers=['o', 'X', 's'], dashes=False, palette=palette)
    # plt.xlim(0, 35)
    plt.tight_layout()
    plt.savefig(f"{base_path}/minimum-runtime-per-iteration.pdf", dpi=300)


def plot_solving_time(optimizer, base_path='./', file_name=None, number_of_experiments=1, min_generation=0, max_generation=70, type="min"):
    data_convergence = []
    data_runtime = []
    for i in range(1, number_of_experiments + 1):
        log = optimizer.load_data_structure(f'{base_path}/data_{i}/log_0.p')
        convergence_factor = log.chapters["convergence_factor"].select(type)
        convergence_factor = convergence_factor[min_generation:max_generation]
        stats = [(gen + 1, tmp, i) for gen, tmp in enumerate(convergence_factor)]
        data_convergence += stats
        minimum_runtime = log.chapters["execution_time"].select(type)
        stats = [(gen + 1, runtime, i) for gen, runtime in enumerate(minimum_runtime) if runtime < 10000]
        data_runtime += stats
    columns_convergence = ['Generation', 'Minimum Convergence Factor', 'Experiment']
    df_convergence = pd.DataFrame(data_convergence, columns=columns_convergence)
    columns_runtime = ['Generation', 'Minimum Execution Time per Iteration (s)', 'Experiment']
    df_runtime = pd.DataFrame(data_runtime, columns=columns_runtime)
    rc('text', usetex=False)
    sns.set_context("paper")
    sns.set_style('ticks', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    sns.despine()
    # palette = sns.color_palette("colorblind", n_colors=3)
    palette = sns.color_palette("colorblind", n_colors=number_of_experiments)
    # g = sns.relplot(x=columns[0], y=columns[1], hue=columns[2],
    #               data=df, kind='line', dashes=False, palette=palette, legend=False)
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(20, 10))
    sns.lineplot(x=columns_convergence[0], y=columns_convergence[1], data=df_convergence, legend=False, ax=ax1)
    sns.lineplot(x=columns_runtime[0], y=columns_runtime[1], data=df_runtime, legend=False, ax=ax2)
    # for ax in g.axes.flatten():
    #   ax.tick_params(axis='y', which='both', direction='out', length=4, left=True)
    #  ax.grid(b=True, which='both', color='gray', linewidth=0.1)
    # ax.yaxis.set_tick_params(which='minor', right='off') 
    # plt.ylim(0, 0.025)
    # plt.title("mu=256,lambda=256,crossover=0.9,pop_init=8")
    # sns.relplot(x=columns[0], y=columns[1], hue=columns[3], style=columns[2],
    #             data=df, kind='line', markers=['o', 'X', 's'], dashes=False, palette=palette)
    # plt.xlim(0, 250)
    plt.tight_layout()
    if file_name is None:
        file_name = f'{base_path}/{type}-objectives.pdf'
    plt.savefig(file_name, dpi=300)


def plot_solving_time_with_variance(base_path='./', file_name=None, start=0, number_of_experiments=10):
    data = []
    elitism_factor = [0.0]
    for start, factor in enumerate(elitism_factor):
        for i in range(number_of_experiments):
            i += start * number_of_experiments
            with open(f'{base_path}/data_{i}/log_0.p', 'rb') as file:
                log = pickle.load(file)
            runtime_avg = log.chapters["fitness"].select('avg')
            runtime_min = log.chapters["fitness"].select('min')
            stats = [(gen + 1, tmp[0] * 1e-3, tmp[1] * 1e-3, i + 1, elitism_factor[start]) for gen, tmp in enumerate(zip(runtime_avg, runtime_min)) if tmp[1] * 1e-3 < 50]
            data += stats

    columns = ['generation', 'avg solving time (s)', 'minimum solving time (s)', 'Experiment', 'elitism']
    df = pd.DataFrame(data, columns=columns)
    rc('text', usetex=True)
    plt.rcParams["font.family"] = "Times New Roman"
    sns.set(style="white")
    sns.set_context("paper")
    sns.despine()
    plt.ylim(0, 50)
    plt.xlim(0, 100)
    plt.tight_layout()
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    plt.subplots()
    fig.suptitle("mu=64,lambda=64,crossover=0.9,pop_init=4,elitism=off", fontsize=16)
    palette = sns.color_palette("colorblind", n_colors=len(elitism_factor))
    # axes[0].set_xlim([0, 50])
    # axes[0].set_ylim([0, 75])
    # axes[1].set_xlim([0, 50])
    # axes[1].set_ylim([0, 12.5])
    axes[0] = sns.lineplot(data=df, x="generation", y="avg solving time (s)", hue="elitism", palette=palette, ax=axes[0], legend=True)
    axes[1] = sns.lineplot(data=df, x="generation", y="minimum solving time (s)", hue="elitism", palette=palette, ax=axes[1], legend=True)

    if file_name is None:
        file_name = f'{base_path}/data_{start}/solving-time.pdf'
    fig.savefig(file_name, quality=100, dpi=300)


def plot_average_solving_time(base_path='./', file_name=None, number_of_experiments=10, min_generation=0, max_generation=250):
    data = []

    def get_total_execution_time(convergence_factor, execution_time_per_iteration):
        infinity = 1e100
        res_reduction_factor = 1e-10
        if convergence_factor < 1:
            return math.log(res_reduction_factor) / math.log(convergence_factor) * execution_time_per_iteration
        else:
            return convergence_factor * math.sqrt(infinity) * execution_time_per_iteration

    for i in range(1, number_of_experiments + 1):
        with open(f'{base_path}/data_{i}/log_0.p', 'rb') as file:
            log = pickle.load(file)
        runtime = log.chapters["execution_time"].select("min")
        convergence_factor = log.chapters["convergence_factor"].select("min")
        stats = [(gen + 1, tmp[1], 1) for gen, tmp in enumerate(zip(runtime, convergence_factor))]
        data += stats
    columns = ['Generation', 'Convergence Factor', 'Experiment']
    df = pd.DataFrame(data, columns=columns)
    rc('text', usetex=True)
    # plt.rcParams["font.family"] = "Times New Roman"
    sns.set_context("paper")
    sns.set_style('ticks', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    # sns.despine()
    # palette = sns.color_palette("colorblind", n_colors=3)
    palette = sns.color_palette("colorblind", n_colors=number_of_experiments)
    a = sns.lineplot(x=columns[0], y=columns[1], data=df)  # hue=columns[2],
    # , kind='line', dashes=False, palette=palette, legend=True)
    # plt.ylim(0, 10000)

    # sns.relplot(x=columns[0], y=columns[1], hue=columns[3], style=columns[2],
    #             data=df, kind='line', markers=['o', 'X', 's'], dashes=False, palette=palette)
    plt.xlim(0, 250)
    plt.ylim(0, 0.4)
    # sns.move_legend(a, loc='upper right')
    plt.tight_layout()
    if file_name is None:
        file_name = "conv-time.pdf"
    plt.savefig(f'{base_path}/{file_name}', dpi=300)


if __name__ == "__main__":
    main()