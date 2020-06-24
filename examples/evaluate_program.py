from evostencils.optimization.program import Optimizer
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
    # settings_path = f'Stokes/2D_FD_Stokes_fromL2.settings'
    # knowledge_path = f'Stokes/2D_FD_Stokes_fromL2.knowledge'

    # 2D Finite difference discretized linear elasticity
    # settings_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.settings'
    # knowledge_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.knowledge'

    # settings_path = f'Helmholtz/2D_FD_Helmholtz_fromL2.settings'
    # knowledge_path = f'Helmholtz/2D_FD_Helmholtz_fromL2.knowledge'
    settings_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.settings'
    knowledge_path = f'Helmholtz/2D_FD_Helmholtz_fromL3.knowledge'
    cycle_name = "VCycle"

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

    infinity = 1e300
    epsilon = 1e-12
    problem_name = program_generator.problem_name

    if not os.path.exists(f'{cwd}/{problem_name}'):
        os.makedirs(f'{cwd}/{problem_name}')
    optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level, equations, operators, fields,
                          program_generator=program_generator,
                          epsilon=epsilon, infinity=infinity)
    maximum_block_size = 8
    with open('grammar_tree.txt', 'r') as file:
        grammar_string = file.read()
    minimum_solver_iterations = 2**3
    maximum_solver_iterations = 2**10
    # krylov_subspace_methods = ('ConjugateGradient', 'BiCGStab', 'MinRes', 'ConjugateResidual')
    krylov_subspace_methods = ()
    optimizer.generate_and_evaluate_program_from_grammar_representation(grammar_string, maximum_block_size,
                                                                        krylov_subspace_methods=krylov_subspace_methods,
                                                                        minimum_solver_iterations=minimum_solver_iterations,
                                                                        maximum_solver_iterations=maximum_solver_iterations,
                                                                        optimize_relaxation_factors=True)


if __name__ == "__main__":
    main()

