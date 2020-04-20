from evostencils.optimization.program import Optimizer
from evostencils.evaluation.convergence import ConvergenceEvaluator
from evostencils.evaluation.performance import PerformanceEvaluator
from evostencils.code_generation.exastencils import ProgramGenerator
from evostencils.initialization.multigrid import generate_operators_from_l2_information
from evostencils.expressions import base, system
from evostencils.optimization.intergrid_transfer import optimize
import os
import lfa_lab
import sys
# import dill
from mpi4py import MPI
# MPI.pickle.__init__(dill.dumps, dill.loads)


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
    # settings_path = f'Stokes/2D_FD_Stokes_fromL2.settings'
    # knowledge_path = f'Stokes/2D_FD_Stokes_fromL2.knowledge'

    # 2D Finite difference discretized linear elasticity
    # settings_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.settings'
    # knowledge_path = f'LinearElasticity/2D_FD_LinearElasticity_fromL2.knowledge'

    # settings_path = f'Helmholtz/2D_FD_Helmholtz_fromL2.settings'
    # knowledge_path = f'Helmholtz/2D_FD_Helmholtz_fromL2.knowledge'

    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    mpi_rank = comm.Get_rank()
    if nprocs > 1:
        tmp = "processes"
    else:
        tmp = "process"
    if mpi_rank == 0:
        print(f"Running {nprocs} MPI {tmp}")
    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path, mpi_rank)

    # Evaluate baseline program
    # program_generator.run_exastencils_compiler()
    # program_generator.run_c_compiler()
    # time, convergence_factor = program_generator.evaluate()
    # print(f'Time: {time}, Convergence factor: {convergence_factor}')

    # Obtain extracted information from program generator
    dimension = program_generator.dimension
    finest_grid = program_generator.finest_grid
    coarsening_factors = program_generator.coarsening_factor
    max_level = program_generator.max_level
    equations = program_generator.equations
    operators = program_generator.operators
    fields = program_generator.fields

    solution_entries = [base.Approximation(f.name, g) for f, g in zip(fields, finest_grid)]
    approximation = system.Approximation('x', solution_entries)
    rhs_entries = [base.RightHandSide(eq.rhs_name, g) for eq, g in zip(equations, finest_grid)]
    rhs = system.RightHandSide('b', rhs_entries)
    coarse_approximation = system.get_coarse_approximation(approximation, coarsening_factors)
    operator, default_restriction, default_prolongation = \
        generate_operators_from_l2_information(equations, operators, fields, max_level, approximation.grid, coarse_approximation.grid)
    restriction, prolongation = optimize(150, program_generator, max_level, operator, default_restriction, default_prolongation,
                                         approximation, coarse_approximation, rhs, operator_range=1)
    prolongation_program = program_generator.generate_prolongation_operator(prolongation.entries[0][0], max_level-1)
    restriction_program = program_generator.generate_restriction_operator(restriction.entries[0][0], max_level)
    print(prolongation_program)
    print(restriction_program)


if __name__ == "__main__":
    main()

