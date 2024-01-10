from evostencils.optimization.program import Optimizer
from evostencils.code_generation.exastencils import ProgramGenerator
import os


def main():
    # TODO adapt to actual path to exastencils project
    dir_name = 'Poisson'
    problem_name = f'2D_FD_{dir_name}_fromL2'
    cwd = os.getcwd()
    compiler_path = f'{cwd}/../exastencils/Compiler/Compiler.jar'
    base_path = f'{cwd}/../exastencils/Examples'

    settings_path = f'{dir_name}/{problem_name}.settings'
    knowledge_path = f'{dir_name}/{problem_name}.knowledge'

    cycle_name= "gen_mgCycle"

    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path, cycle_name=cycle_name)

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

    if not os.path.exists(f'{cwd}/{problem_name}'):
        os.makedirs(f'{cwd}/{problem_name}')
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                          program_generator=program_generator,
                          epsilon=epsilon, infinity=infinity)
    maximum_block_size = 8
    for i in range(0, 10):
        print(f"data_{i}")
        best_solving_time = infinity
        best_convergence_factor = infinity
        best_number_of_iterations = infinity
        best_index = 0
        for j in range(0, 20):
            with open(f'../gpem-21-results/{problem_name}/data_{i}/hof_0/individual_{j}.txt', 'r') as file:
                grammar_string = file.read()
            # print(f"Individual {j}")
            solving_time, convergence_factor, number_of_iterations = optimizer.generate_and_evaluate_program_from_grammar_representation(grammar_string, maximum_block_size)
            if solving_time < best_solving_time:
                best_solving_time = solving_time
                best_convergence_factor = convergence_factor
                best_number_of_iterations = number_of_iterations
                best_index = j
        print("Fastest solver: Individual ", best_index, flush=True)
        print(f'Solving Time: {best_solving_time}, '
              f'Convergence factor: {best_convergence_factor}, '
              f'Number of Iterations: {best_number_of_iterations}', flush=True)


if __name__ == "__main__":
    main()

