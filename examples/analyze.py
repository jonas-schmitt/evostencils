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

    file_name = f'{problem_name}/data'

    pop0 = optimizer.load_data_structure(f'{file_name}/pop_0.p')
    log0 = optimizer.load_data_structure(f'{file_name}/log_0.p')
    hof0 = tools.ParetoFront(similar=lambda a, b: a.fitness == b.fitness)
    hof0.update(pop0)

    pop1 = optimizer.load_data_structure(f'{file_name}/pop_1.p')
    log1 = optimizer.load_data_structure(f'{file_name}/log_1.p')
    hof1 = tools.ParetoFront(similar=lambda a, b: a.fitness == b.fitness)
    hof1.update(pop1)

    sns.set(style="darkgrid")
    front0 = [(*ind.fitness.values, 0) for ind in hof0]
    front1 = [(*ind.fitness.values, 1) for ind in hof1]
    front = front0 + front1

    tips = sns.load_dataset("tips")
    # x = front[:, 0]
    # y = front[:, 1]
    # coeffs = np.polyfit(x, y, 3)
    # x_fit = np.linspace(x[0], x[-1], num=len(x)*10)
    # y_fit = np.polyval(coeffs, x_fit)
    columns = ['Spectral Radius', 'Runtime per Iteration (ms)', 'Run']
    df = pd.DataFrame(front, columns=columns)
    sns.relplot(x=columns[0], y=columns[1], hue=columns[2], data=df, kind='scatter')
    # plt.plot(x_fit, y_fit)
    plt.show()

    #plt.scatter(x, y, c="b")
    #plt.plot(x_fit, y_fit)
    #plt.axis("tight")
    #plt.show()


if __name__ == "__main__":
    main()

