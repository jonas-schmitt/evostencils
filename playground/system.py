from evostencils.code_generation.exastencils import ProgramGenerator
from evostencils.initialization import multigrid, parser
from evostencils.expressions import base, system, smoother, partitioning as part
import os
import pickle
import lfa_lab

cwd = os.getcwd()
compiler_path = f'{cwd}/../exastencils/Compiler/compiler.jar'
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

# 2D Finite difference discretized biharmonic equation
# settings_path = f'BiHarmonic/2D_FD_BiHarmonic_fromL2.settings'
# knowledge_path = f'BiHarmonic/2D_FD_BiHarmonic_fromL2.knowledge'

# 2D Finite difference discretized Stokes
# settings_path = f'Stokes/2D_FD_Stokes_fromL2.settings'
# knowledge_path = f'Stokes/2D_FD_Stokes_fromL2.knowledge'


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
coarse_grid = system.get_coarse_grid(finest_grid, coarsening_factors)
min_level = program_generator.min_level
max_level = program_generator.max_level
equations = program_generator.equations
operators = program_generator.operators
fields = program_generator.fields

block_sizes = [(1, 1), (2, 1), (2, 2)]

operator, restriction, prolongation, = \
    multigrid.generate_operators_from_l2_information(equations, operators, fields, max_level, finest_grid, coarse_grid)
solution_entries = [base.Approximation(f.name, g) for f, g in zip(fields, finest_grid)]
approximation = system.Approximation('x', solution_entries)
rhs_entries = [base.RightHandSide(eq.rhs_name, g) for eq, g in zip(equations, finest_grid)]
rhs = system.RightHandSide('b', rhs_entries)
residual = base.Residual(operator, approximation, rhs)
tmp = base.Inverse(smoother.generate_collective_block_jacobi(operator, block_sizes))
new_correction = base.Multiplication(tmp, residual)
cycle = base.Cycle(approximation, rhs, new_correction, partitioning=part.RedBlack, relaxation_factor=0.8)
new_approximation = cycle
residual = base.Residual(operator, new_approximation, rhs)
tmp = base.Multiplication(restriction, residual)
coarse_operator, coarse_restriction, coarse_prolongation, = \
    multigrid.generate_operators_from_l2_information(equations, operators, fields, max_level-1, coarse_grid, system.get_coarse_grid(coarse_grid, coarsening_factors))
cgs = base.CoarseGridSolver(coarse_operator)
tmp = base.Multiplication(cgs, tmp)
tmp = base.Multiplication(prolongation, tmp)
cycle = base.Cycle(approximation, rhs, tmp)
new_approximation = cycle
new_residual = base.Residual(operator, new_approximation, rhs)
tmp = base.Inverse(smoother.generate_collective_block_jacobi(operator, block_sizes))
new_correction = base.Multiplication(tmp, new_residual)
new_cycle = base.Cycle(new_approximation, rhs, new_correction, partitioning=part.RedBlack, relaxation_factor=0.8)
new_approximation = new_cycle
new_residual = base.Residual(operator, new_approximation, rhs)
tmp = base.Inverse(smoother.generate_collective_block_jacobi(operator, block_sizes))
new_correction = base.Multiplication(tmp, new_residual)
new_cycle = base.Cycle(new_approximation, rhs, new_correction, partitioning=part.RedBlack, relaxation_factor=0.8)
storages = program_generator.generate_storage(min_level, max_level, finest_grid)
cycle_function = program_generator.generate_cycle_function(new_cycle, storages, max_level - 1, max_level)
print(cycle_function)
program_generator.generate_level_adapted_knowledge_file(max_level)
program_generator.run_exastencils_compiler()
program_generator.generate_l3_file(cycle_function)
program_generator.run_c_compiler()
time, convergence_factor = program_generator.evaluate(number_of_samples=1)
print(f'Time: {time}, Convergence factor: {convergence_factor}')
program_generator.restore_files()
