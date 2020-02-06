from evostencils.optimization.program import Optimizer
from evostencils.evaluation.convergence import ConvergenceEvaluator
from evostencils.evaluation.performance import PerformanceEvaluator
from evostencils.code_generation.exastencils import ProgramGenerator
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import lfa_lab
import numpy as np
from deap import tools
from math import log
from matplotlib import rc


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
    # settings_path = f'Stokes/2D_FV_Stokes_fromL2.settings'
    # knowledge_path = f'Stokes/2D_FV_Stokes_fromL2.knowledge'

    # 2D Finite difference discretized linear elasticity
    # settings_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.settings'
    # knowledge_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.knowledge'

    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path)

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
    peak_performance = 20344.07 * 1e6
    peak_bandwidth = 19255.70 * 1e6
    performance_evaluator = PerformanceEvaluator(peak_performance, peak_bandwidth, bytes_per_word)
    infinity = np.finfo(np.float64).max
    epsilon = 1e-10
    problem_name = program_generator.problem_name

    if not os.path.exists(problem_name):
        os.makedirs(problem_name)
    checkpoint_directory_path = f'{problem_name}/checkpoints'
    if not os.path.exists(checkpoint_directory_path):
        os.makedirs(checkpoint_directory_path)
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                          convergence_evaluator=convergence_evaluator,
                          performance_evaluator=performance_evaluator, program_generator=program_generator,
                          epsilon=epsilon, infinity=infinity, checkpoint_directory_path=checkpoint_directory_path)
    base_path = ''
    hofs = []
    front = []
    for i in range(0, 3):
        file_name = f'{base_path}_NSGAII_{i}/data'
        pop = optimizer.load_data_structure(f'{file_name}/pop_0.p')
        hof = tools.ParetoFront(similar=lambda a, b: a.fitness == b.fitness)
        hof.update(pop)
        hofs.append(hof)
        front += [(*ind.fitness.values, 'NSGA-II', i+1) for ind in hof]

    for i in range(0, 3):
        file_name = f'{base_path}_RANDOM_{i}/data'
        pop = optimizer.load_data_structure(f'{file_name}/pop_0.p')
        hof = tools.ParetoFront(similar=lambda a, b: a.fitness == b.fitness)
        hof.update(pop)
        hofs.append(hof)
        front += [(*ind.fitness.values, 'Random', i+1) for ind in hof]

    columns = ['Estimated Number of Iterations', 'Estimated Runtime per Iteration (ms)', 'Method', 'Experiment']
    df = pd.DataFrame(front, columns=columns)
    rc('text', usetex=True)
    sns.set(style="white")
    sns.set_context("paper")
    sns.despine()
    palette = sns.color_palette("colorblind", n_colors=3)
    sns.relplot(x=columns[0], y=columns[1], hue=columns[3], style=columns[2],
                data=df, kind='line', markers=['o', 'X', 's'], dashes=False, palette=palette)
    plt.xlim(0, 35)
    plt.tight_layout()
    plt.savefig("pareto-front.pdf", quality=100, dpi=300)


if __name__ == "__main__":
    main()

