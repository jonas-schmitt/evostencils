from evostencils.code_generation.exastencils import ProgramGenerator
import os


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

    # 2D Finite difference discretized linear elasticity
    settings_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.settings'
    knowledge_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.knowledge'

    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path)
    program_generator.run_c_compiler()
    time_to_solution, convergence_factor = program_generator.evaluate(number_of_samples=10)
    print(f'Time to solution: {time_to_solution} ms, Convergence Factor: {convergence_factor}')


if __name__ == "__main__":
    main()

