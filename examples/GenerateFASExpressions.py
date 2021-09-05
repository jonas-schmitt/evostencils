import os

from evostencils.expressions import base, system
from evostencils.expressions.reference_cycles import generate_v_22_cycle_two_grid, generate_FAS_v_22_cycle_two_grid
from evostencils.initialization.multigrid import Terminals, generate_primitive_set
from evostencils.expressions.base import Grid, sub
from evostencils.expressions.base import Operator as baseOperator
from evostencils.expressions.base import Restriction as baseRestriction
from evostencils.expressions.base import Prolongation as baseProlongation
from evostencils.expressions.base import RightHandSide as baseRHS
from evostencils.expressions.system import RightHandSide, ZeroApproximation, Operator, Restriction, Prolongation, get_coarse_grid, get_coarse_operator
from evostencils.stencils.gallery import Poisson2D
from evostencils.types import level_control
from examples.plot_computational_graph import viz, save
from evostencils.code_generation.exastencils import ProgramGenerator



generate = "PrimitiveSet"

if generate == "GraphRepresentation":
    fg = [Grid([8, 8], [1.0 / 8, 1.0 / 8], 3)]
    approximation = ZeroApproximation(fg, 'u')
    entries = [[baseOperator('A', g, Poisson2D()) for g in fg] for _ in fg]
    operator = Operator('A', entries)
    dimension = 2
    coarsening_factor = [(2, 2)]
    cg = get_coarse_grid(fg, coarsening_factor)
    coarse_operator = get_coarse_operator(operator, cg)
    list_restriction_operators = [baseRestriction('I_h_2h', g, gc, baseOperator('A', g, Poisson2D()).stencil_generator)  # Omit stencil generator for now
                                  for g, gc in zip(fg, cg)]
    list_prolongation_operators = [baseProlongation('I_2h_h', g, gc, baseOperator('A', g, Poisson2D()).stencil_generator)
                                   for g, gc in zip(fg, cg)]
    restriction = Restriction('I_h_2h', list_restriction_operators)
    prolongation = Prolongation('I_2h_h', list_prolongation_operators)
    terminals_fine_level = Terminals(approximation, dimension, coarsening_factor, operator, coarse_operator, restriction, prolongation)
    rhs = RightHandSide('f', [baseRHS('f', g) for g in fg])

    U1 = generate_v_22_cycle_two_grid(terminals_fine_level, rhs)
    U2 = generate_FAS_v_22_cycle_two_grid(terminals_fine_level, rhs)

    print(U1)
    print(U1.correction)
    print(U1.approximation)
    node = 0
    # viz(node, U1)
    save()
elif generate == "PrimitiveSet":

    # ExaStencils configuration
    #dir_name = 'Helmholtz'
    dir_name = 'LinearElasticity'
    problem_name = f'2D_FD_{dir_name}_fromL2'
    cwd = os.getcwd()
    compiler_path = f'{cwd}/../../exastencils/Compiler/Compiler.jar'
    base_path = f'{cwd}/../../exastencils/Examples'
    settings_path = f'{dir_name}/{problem_name}.settings'
    knowledge_path = f'{dir_name}/{problem_name}.knowledge'
    cycle_name = "gen_mgCycle"
    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path, cycle_name=cycle_name)

    dimension = program_generator.dimension
    finest_grid = program_generator.finest_grid
    coarsening_factor = program_generator.coarsening_factor
    min_level = program_generator.min_level
    max_level = program_generator.max_level
    equations = program_generator.equations
    operators = program_generator.operators
    fields = program_generator.fields
    solution_entries = [base.Approximation(f.name, g) for f, g in zip(fields, finest_grid)]
    approximation = system.Approximation('x', solution_entries)
    rhs_entries = [base.RightHandSide(eq.rhs_name, g) for eq, g in zip(equations, finest_grid)]
    rhs = system.RightHandSide('b', rhs_entries)
    maximum_block_size = 8
    levels_per_run = max_level - min_level

    pset, terminal_list = generate_primitive_set(approximation, rhs, dimension,
                                                 coarsening_factor, max_level, equations,
                                                 operators, fields,
                                                 maximum_block_size=maximum_block_size,
                                                 depth=levels_per_run,
                                                 LevelFinishedType=level_control.generate_finished_type(),
                                                 LevelNotFinishedType=level_control.generate_not_finished_type())

    print("completed")