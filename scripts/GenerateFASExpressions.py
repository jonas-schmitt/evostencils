import os

from evostencils.expressions import base, system
from evostencils.expressions.reference_cycles import generate_v_22_cycle_three_grid, generate_FAS_v_22_cycle_two_grid, generate_v_22_cycle_two_grid, generate_FAS_v_22_cycle_three_grid
from evostencils.initialization.multigrid import Terminals, generate_primitive_set
from evostencils.expressions.base import Grid, sub
from evostencils.expressions.base import Operator as baseOperator
from evostencils.expressions.base import Restriction as baseRestriction
from evostencils.expressions.base import Prolongation as baseProlongation
from evostencils.expressions.base import RightHandSide as baseRHS
from evostencils.expressions.system import RightHandSide, ZeroApproximation, Operator, Restriction, Prolongation, get_coarse_grid, get_coarse_operator
from evostencils.stencils.gallery import Poisson2D
from evostencils.types import level_control
from scripts.plot_computational_graph import create_graph, save_graph
from evostencils.code_generation.exastencils import ProgramGenerator
from plot_computational_graph import get_sub, get_super
from deap import creator, gp, tools
from deap import base as deap_base
from evostencils.genetic_programming import genGrow

generate = "PrimitiveSet"


def visualize_tree(expression, filename):
    import pygraphviz as pgv
    nodes, edges, labels = gp.graph(expression)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")
    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]
    g.draw(f"{filename}.png", "png")


def init_toolbox(ipset):
    creator.create("SingleObjectiveFitness", deap_base.Fitness, weights=(-1.0,))
    creator.create("SingleObjectiveIndividual", gp.PrimitiveTree, fitness=creator.SingleObjectiveFitness)
    toolbox = deap_base.Toolbox()
    toolbox.register("expression", genGrow, pset=ipset, min_height=0, max_height=15)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("individual", tools.initIterate, creator.SingleObjectiveIndividual, toolbox.expression)
    return toolbox


if generate == "GraphRepresentation":
    fg = [Grid([8, 8], [1.0 / 8, 1.0 / 8], 3)]
    coarsening_factor = [(2, 2)]
    cg = get_coarse_grid(fg, coarsening_factor)
    ccg = get_coarse_grid(cg, coarsening_factor)
    approximation = ZeroApproximation(fg, 'u')
    approximation_c = ZeroApproximation(cg, 'u')

    entries = [[baseOperator('A', g, Poisson2D()) for g in fg] for _ in fg]
    operator = Operator('A', entries)
    dimension = 2
    coarse_operator = get_coarse_operator(operator, cg)
    cc_operator = get_coarse_operator(operator, ccg)
    list_restriction_operators = [baseRestriction('I_h_2h', g, gc, baseOperator('A', g, Poisson2D()).stencil_generator)  # Omit stencil generator for now
                                  for g, gc in zip(fg, cg)]
    list_prolongation_operators = [baseProlongation('I_2h_h', g, gc, baseOperator('A', g, Poisson2D()).stencil_generator)
                                   for g, gc in zip(fg, cg)]
    restriction = Restriction(f'I{get_sub("a")}{get_super("2a")}', list_restriction_operators)
    prolongation = Prolongation(f'I{get_sub("2a")}{get_super("a")}', list_prolongation_operators)
    list_restriction_operators_c = [baseRestriction('I_2h_4h', g, gc, baseOperator('A', g, Poisson2D()).stencil_generator)  # Omit stencil generator for now
                                    for g, gc in zip(cg, ccg)]
    list_prolongation_operators_c = [baseProlongation('I_4h_2h', g, gc, baseOperator('A', g, Poisson2D()).stencil_generator)
                                     for g, gc in zip(cg, ccg)]
    restriction_c = Restriction(f'I{get_sub("2a")}{get_super("4a")}', list_restriction_operators_c)
    prolongation_c = Prolongation(f'I{get_sub("4a")}{get_super("2a")}', list_prolongation_operators_c)
    terminals_fine_level = Terminals(approximation, dimension, coarsening_factor, operator, coarse_operator, restriction, prolongation)
    terminals_coarse = Terminals(approximation_c, dimension, coarsening_factor, coarse_operator, cc_operator, restriction_c, prolongation_c)
    rhs = RightHandSide('f', [baseRHS('f', g) for g in fg])

    U1 = generate_FAS_v_22_cycle_three_grid(terminals_fine_level, terminals_coarse, rhs)  # terminals_coarse, rhs)
    U2 = generate_FAS_v_22_cycle_two_grid(terminals_fine_level, rhs)
    print(U1)
    print(U1.correction)
    print(U1.approximation)
    create_graph(U1)
    save_graph()
elif generate == "PrimitiveSet":

    # ExaStencils configuration
    # dir_name = 'Helmholtz'
    dir_name = 'LinearElasticity'
    problem_name = f'2D_FD_{dir_name}_fromL2'
    cwd = os.getcwd()
    compiler_path = f'{cwd}/../../exastencils/Compiler/Compiler.jar'
    base_path = f'{cwd}/../example_problems'
    platform_path = f'lib/linux.platform'
    settings_path = f'{dir_name}/{problem_name}.settings'
    knowledge_path = f'{dir_name}/{problem_name}.knowledge'
    cycle_name = "gen_mgCycle"
    program_generator = ProgramGenerator(compiler_path, base_path, settings_path, knowledge_path, platform_path, cycle_name=cycle_name)

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
                                                 maximum_local_system_size=maximum_block_size,
                                                 depth=levels_per_run,
                                                 LevelFinishedType=level_control.generate_finished_type(),
                                                 LevelNotFinishedType=level_control.generate_not_finished_type())
    toolbox = init_toolbox(pset)
    expr = toolbox.individual()

    obj = gp.compile(expr, pset)
    create_graph(obj[0])
    save_graph()
    # visualize_tree(expr, "graph")
print("completed")
