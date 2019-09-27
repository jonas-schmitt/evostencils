import sys, os
from evostencils.code_generation.exastencils import ProgramGenerator


def main():

    cwd = os.getcwd()
    compiler_path = f'{cwd}/../exastencils/Compiler/compiler.jar'
    base_path = f'{cwd}/../exastencils/Examples'
    settings_path = f'BiHarmonic/2D_FD_BiHarmonic_fromL2.settings'
    knowledge_path = f'BiHarmonic/2D_FD_BiHarmonic_fromL2.knowledge'
    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path)
    program_generator.run_c_compiler()
    time_to_solution, convergence_factor = program_generator.evaluate(number_of_samples=10)
    print(f'Time to solution: {time_to_solution} ms, Convergence Factor: {convergence_factor}')


if __name__ == "__main__":
    main()

