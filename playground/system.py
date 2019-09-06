from evostencils.code_generation.exastencils import ProgramGenerator
from evostencils.initialization import multigrid, parser
from evostencils.expressions import base, system, transformations
import os
import pickle
import lfa_lab

cwd = os.getcwd()
compiler_path = f'{cwd}/../exastencils/Compiler/compiler.jar'
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

# 2D Finite difference discretized biharmonic equation
settings_path = f'BiHarmonic/2D_FD_BiHarmonic_fromL2.settings'
knowledge_path = f'BiHarmonic/2D_FD_BiHarmonic_fromL2.knowledge'

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

operator, restriction, prolongation, = \
    multigrid.generate_operators_from_l2_information(equations, operators, fields, max_level, finest_grid, coarse_grid)
solution_entries = [base.Approximation(f.name, g) for f, g in zip(fields, finest_grid)]
approximation = system.Approximation('x', solution_entries)
rhs_entries = [base.RightHandSide(eq.rhs_name, g) for eq, g in zip(equations, finest_grid)]
rhs = system.RightHandSide('b', rhs_entries)
residual = base.Residual(operator, approximation, rhs)
tmp = base.Multiplication(restriction, residual)
coarse_operator, coarse_restriction, coarse_prolongation, = \
    multigrid.generate_operators_from_l2_information(equations, operators, fields, max_level-1, coarse_grid, system.get_coarse_grid(coarse_grid, coarsening_factors))
cgs = base.CoarseGridSolver(coarse_operator)
tmp = base.Multiplication(cgs, tmp)
tmp = base.Multiplication(prolongation, tmp)
cycle = base.Cycle(approximation, rhs, tmp)
storages = program_generator.generate_storage(min_level, max_level, finest_grid)
program = program_generator.generate_cycle_function(cycle, storages, max_level-1, max_level)
tmp = transformations.obtain_sympy_expression_for_local_system(system.Diagonal(operator), operator, equations, fields)
for k, v in tmp.items():
    print(k, v[0].free_symbols)
print(program)
