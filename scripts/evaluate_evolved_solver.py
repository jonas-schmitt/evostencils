from evostencils.optimization.program import Optimizer
from evostencils.code_generation.exastencils import ProgramGenerator
import os
import evostencils

def main():
    # TODO adapt to actual path to exastencils project
    dir_name = 'Poisson'
    problem_name = f'2D_FD_{dir_name}_fromL2'
    cwd = os.path.dirname(evostencils.__file__)
    compiler_path = f'{cwd}/../exastencils/Compiler/Compiler.jar'
    base_path = f'{cwd}/../exastencils/Examples'

    settings_path = f'{dir_name}/{problem_name}.settings'
    knowledge_path = f'{dir_name}/{problem_name}.knowledge'
    platform_path = f'lib/linux.platform'
    cycle_name= "gen_mgCycle"

    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path, cycle_name=cycle_name, platform_path=platform_path)

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
    path_to_individual = "" # TODO insert path to individuals
    # with open(path_to_individual, 'r') as file:
    with open(f'{problem_name}/data_0/hof_0/individual_0.txt', 'r') as file:
        grammar_string = file.read()
    # print(f"Individual {j}")
    solving_time, convergence_factor, number_of_iterations = optimizer.generate_and_evaluate_program_from_grammar_representation(grammar_string, maximum_block_size)
    print(f'Solving Time: {solving_time}, '
          f'Convergence factor: {convergence_factor}, '
          f'Number of Iterations: {number_of_iterations}', flush=True)


if __name__ == "__main__":
    main()

