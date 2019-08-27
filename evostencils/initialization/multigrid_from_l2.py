from evostencils.expressions import multigrid as mg
from evostencils.expressions import base
from evostencils.expressions import system
from evostencils.expressions import partitioning as part
from evostencils.types import operator as matrix_types
from evostencils.types import grid as grid_types
from evostencils.types import multiple
from evostencils.types import partitioning, level_control
from evostencils.deap_extension import PrimitiveSetTyped
from deap import gp
from evostencils.code_generation import parser
import sympy


class ConstantStencilGenerator:
    def __init__(self, stencil):
        self._stencil = stencil

    def generate_stencil(self, _):
        return self._stencil

def generate_operator_entries_from_equation(equation, operators: list, fields, grid):
    row_of_operators = []
    indices = []

    def recursive_descent(expr):
        if expr.func == sympy.Add:
            for arg in expr.args:
                recursive_descent(arg)
        elif expr.func == sympy.Mul and len(expr.args) == 2 and expr.args[-1] in fields:
            op_symbol = expr.args[0]
            field_symbol = expr.args[1]
            field_index = fields.index(field_symbol)
            j = next(k for k, op_info in enumerate(operators) if op_symbol.name == op_info.name)
            row_of_operators.append(base.Operator(op_symbol.name, grid[field_index], ConstantStencilGenerator(operators[j].stencil)))
            indices.append(field_index)
    recursive_descent(equation.sympy_expr)
    for i in range(len(grid)):
        if i not in indices:
            row_of_operators.append(base.ZeroOperator(grid[i]))
            indices.append(i)
    result = [operator for (index, operator) in sorted(zip(indices, row_of_operators), key=lambda p: p[0])]
    return result


def generate_system_operator_from_l2_information(equations: [parser.EquationInfo], operators: [parser.OperatorInfo],
                                               fields: [sympy.Symbol], level, grid: [base.Grid]):
    operators_on_level = list(filter(lambda x: x.level == level, operators))
    system_operators = []
    for op_info in operators_on_level:
        if op_info.operator_type != mg.Restriction and op_info.operator_type != mg.Prolongation:
            system_operators.append(op_info)
    entries = []
    for equation in equations:
        row_of_entries = generate_operator_entries_from_equation(equation, system_operators, fields, grid)
        entries.append(row_of_entries)

    operator = system.Operator(f'A_{level}', entries)

    return operator


def generate_operators_from_l2_information(equations: [parser.EquationInfo], operators: [parser.OperatorInfo],
                                           fields: [sympy.Symbol], level, fine_grid: [base.Grid], coarse_grid: [base.Grid]):
    operators_on_level = list(filter(lambda x: x.level == level, operators))
    restriction_operators = []
    prolongation_operators = []
    system_operators = []
    for op_info in operators_on_level:
        if op_info.operator_type == mg.Restriction:
            restriction_operators.append(op_info)
        elif op_info.operator_type == mg.Prolongation:
            prolongation_operators.append(op_info)
        else:
            system_operators.append(op_info)
    assert len(restriction_operators) == len(fields), 'The number of restriction operators does not match with the number of fields'
    assert len(prolongation_operators) == len(fields), 'The number of prolongation operators does not match with the number of fields'
    list_of_stencil_generators = [ConstantStencilGenerator(op_info.stencil) for i, op_info in enumerate(restriction_operators)]
    restriction = system.Restriction(f'R_{level}', fine_grid, coarse_grid, list_of_stencil_generators)

    list_of_stencil_generators = [ConstantStencilGenerator(op_info.stencil) for i, op_info in enumerate(prolongation_operators)]
    prolongation = system.Prolongation(f'P_{level}', fine_grid, coarse_grid, list_of_stencil_generators)

    entries = []
    for equation in equations:
        row_of_entries = generate_operator_entries_from_equation(equation, system_operators, fields, fine_grid)
        entries.append(row_of_entries)

    operator = system.Operator(f'A_{level}', entries)

    return operator, restriction, prolongation


class Terminals:
    def __init__(self, approximation, dimension, coarsening_factors, operator, coarse_operator, restriction, prolongation,
                 coarse_grid_solver_expression=None):
        self.operator = operator
        self.approximation = approximation
        self.grid = approximation.grid
        self.dimension = dimension
        self.coarsening_factor = coarsening_factors
        self.prolongation = prolongation
        self.restriction = restriction
        self.coarse_grid = system.get_coarse_grid(self.approximation.grid, self.coarsening_factor)
        self.coarse_operator = coarse_operator
        self.identity = system.Identity(approximation.grid)
        self.coarse_grid_solver = mg.CoarseGridSolver(self.coarse_operator, expression=coarse_grid_solver_expression)
        self.no_partitioning = part.Single
        self.red_black_partitioning = part.RedBlack


class Types:
    def __init__(self, terminals: Terminals, FinishedType, NotFinishedType):
        self.Operator = matrix_types.generate_operator_type(terminals.operator.shape)
        types = [grid_types.generate_grid_type(grid.size) for grid in terminals.grid]
        self.Grid = multiple.generate_type_list(*types)
        types = [grid_types.generate_correction_type(grid.size) for grid in terminals.grid]
        self.Correction = multiple.generate_type_list(*types)
        types = [grid_types.generate_rhs_type(grid.size) for grid in terminals.grid]
        self.RHS = multiple.generate_type_list(*types)
        self.DiagonalOperator = matrix_types.generate_diagonal_operator_type(terminals.operator.shape)
        self.BlockDiagonalOperator = matrix_types.generate_block_diagonal_operator_type(terminals.operator.shape)
        self.Prolongation = matrix_types.generate_operator_type(terminals.prolongation.shape)
        self.Restriction = matrix_types.generate_operator_type(terminals.restriction.shape)
        self.CoarseOperator = matrix_types.generate_operator_type(terminals.coarse_operator.shape)
        self.CoarseGridSolver = matrix_types.generate_solver_type(terminals.coarse_operator.shape)
        types = [grid_types.generate_grid_type(grid.size) for grid in terminals.coarse_grid]
        self.CoarseGrid = multiple.generate_type_list(*types)
        types = [grid_types.generate_rhs_type(grid.size) for grid in terminals.coarse_grid]
        self.CoarseRHS = multiple.generate_type_list(*types)
        types = [grid_types.generate_correction_type(grid.size) for grid in terminals.coarse_grid]
        self.CoarseCorrection = multiple.generate_type_list(*types)
        self.Partitioning = partitioning.generate_any_partitioning_type()
        self.Finished = FinishedType
        self.NotFinished = NotFinishedType


def add_cycle(pset: gp.PrimitiveSetTyped, terminals: Terminals, types: Types, level, coarsest=False):
    null_grid_coarse = system.ZeroApproximation(terminals.coarse_grid)
    pset.addTerminal(null_grid_coarse, types.CoarseGrid, f'zero_grid_{level+1}')
    pset.addTerminal(base.inv(system.Diagonal(terminals.operator)), types.DiagonalOperator, f'D_inv_{level}')
    pset.addTerminal(terminals.prolongation, types.Prolongation, f'P_{level}')
    pset.addTerminal(terminals.restriction, types.Restriction, f'R_{level}')

    OperatorType = types.Operator
    GridType = types.Grid
    CorrectionType = types.Correction

    def coarse_cycle(coarse_grid, cycle, partitioning):
        result = mg.cycle(cycle.approximation, cycle.rhs,
                          mg.cycle(coarse_grid, cycle.correction, mg.residual(terminals.coarse_operator, coarse_grid, cycle.correction),
                                   partitioning), cycle.partitioning, predecessor=cycle.predecessor)
        result.correction.predecessor = result
        return result.correction

    def residual(args, partitioning):
        return mg.cycle(args[0], args[1], mg.residual(terminals.operator, args[0], args[1]), partitioning, predecessor=args[0].predecessor)

    def apply(operator, cycle):
        return mg.cycle(cycle.approximation, cycle.rhs, base.mul(operator, cycle.correction), cycle.partitioning, cycle.weight,
                        cycle.predecessor)

    def coarse_grid_correction(interpolation, cycle):
        cycle.predecessor._correction = base.mul(interpolation, cycle)
        return iterate(cycle.predecessor)

    def iterate(cycle):
        from evostencils.expressions import transformations
        new_cycle = transformations.repeat(cycle, 1)
        return new_cycle, new_cycle.rhs

    pset.addPrimitive(iterate, [multiple.generate_type_list(types.Grid, types.Correction, types.Finished)], multiple.generate_type_list(types.Grid, types.RHS, types.Finished), f"iterate_{level}")
    pset.addPrimitive(iterate, [multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished)], multiple.generate_type_list(types.Grid, types.RHS, types.NotFinished), f"iterate_{level}")

    pset.addPrimitive(residual, [multiple.generate_type_list(types.Grid, types.RHS, types.Finished), types.Partitioning], multiple.generate_type_list(GridType, CorrectionType, types.Finished), f"residual_{level}")
    pset.addPrimitive(residual, [multiple.generate_type_list(types.Grid, types.RHS, types.NotFinished), types.Partitioning], multiple.generate_type_list(GridType, CorrectionType, types.NotFinished), f"residual_{level}")
    pset.addPrimitive(apply, [OperatorType, multiple.generate_type_list(types.Grid, types.Correction, types.Finished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.Finished),
                      f"apply_{level}")
    pset.addPrimitive(apply, [OperatorType, multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished),
                      f"apply_{level}")
    if not coarsest:

        pset.addPrimitive(coarse_grid_correction, [types.Prolongation, multiple.generate_type_list(types.CoarseGrid, types.CoarseCorrection, types.Finished)], multiple.generate_type_list(types.Grid, types.RHS, types.Finished), f"cgc_{level}")

        pset.addPrimitive(coarse_cycle,
                          [types.CoarseGrid, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.NotFinished),
                           types.Partitioning],
                          multiple.generate_type_list(types.CoarseGrid, types.CoarseCorrection, types.NotFinished),
                          f"coarse_cycle_{level}")
        pset.addPrimitive(coarse_cycle,
                          [types.CoarseGrid, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished),
                           types.Partitioning],
                          multiple.generate_type_list(types.CoarseGrid, types.CoarseCorrection, types.Finished),
                          f"coarse_cycle_{level}")

    else:
        def solve(cgs, interpolation, cycle):
            new_cycle = mg.cycle(cycle.approximation, cycle.rhs, base.mul(interpolation, base.mul(cgs, cycle.correction)), cycle.partitioning, cycle.weight,
                                 cycle.predecessor)
            return iterate(new_cycle)

        pset.addPrimitive(solve, [types.CoarseGridSolver, types.Prolongation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.NotFinished)],
                          multiple.generate_type_list(types.Grid, types.RHS, types.Finished),
                          f'solve_{level}')
        pset.addPrimitive(solve, [types.CoarseGridSolver, types.Prolongation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished)],
                          multiple.generate_type_list(types.Grid, types.RHS, types.Finished),
                          f'solve_{level}')

        pset.addTerminal(terminals.coarse_grid_solver, types.CoarseGridSolver, f'S_{level}')
        """
        pset.addPrimitive(extend, [types.CoarseGridSolver, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelNotFinished)],
                          multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelFinished),
                          f'solve_{level}')
        pset.addPrimitive(extend, [types.CoarseGridSolver, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelFinished)],
                          multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelFinished),
                          f'solve_{level}')

        pset.addPrimitive(extend, [types.Interpolation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelFinished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.LevelFinished),
                      f'interpolate_{level}')
        pset.addPrimitive(extend, [types.Interpolation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelNotFinished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.LevelNotFinished),
                      f'interpolate_{level}')
        """
    # Multigrid recipes
    pset.addPrimitive(apply, [types.Restriction, multiple.generate_type_list(types.Grid, types.Correction, types.Finished)],
                      multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished),
                      f'restrict_{level}')
    pset.addPrimitive(apply, [types.Restriction, multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished)],
                      multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.NotFinished),
                      f'restrict_{level}')
    """
    pset.addPrimitive(extend, [types.Interpolation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelFinished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.LevelFinished),
                      f'interpolate_{level}')
    pset.addPrimitive(extend, [types.Interpolation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelNotFinished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.LevelNotFinished),
                      f'interpolate_{level}')
    """


def generate_primitive_set(approximation, rhs, dimension, coarsening_factors, max_level, equations, operators, fields,
                           coarse_grid_solver_expression=None, depth=2, LevelFinishedType=None, LevelNotFinishedType=None):
    assert depth >= 1, "The maximum number of cycles must be greater zero"
    coarsest = False
    cgs_expression = None
    if depth == 1:
        coarsest = True
        cgs_expression = coarse_grid_solver_expression
    fine_grid = approximation.grid
    coarse_grid = system.get_coarse_grid(fine_grid, coarsening_factors)
    operator, restriction, prolongation, = \
        generate_operators_from_l2_information(equations, operators, fields, max_level, fine_grid, coarse_grid)
    coarse_operator, coarse_restriction, coarse_prolongation, = \
        generate_operators_from_l2_information(equations, operators, fields, max_level-1, coarse_grid, system.get_coarse_grid(coarse_grid, coarsening_factors))
    terminals = Terminals(approximation, dimension, coarsening_factors, operator, coarse_operator, restriction, prolongation, cgs_expression)
    if LevelFinishedType is None:
        LevelFinishedType = level_control.generate_finished_type()
    if LevelNotFinishedType is None:
        LevelNotFinishedType = level_control.generate_not_finished_type()
    types = Types(terminals, LevelFinishedType, LevelNotFinishedType)
    pset = PrimitiveSetTyped("main", [], multiple.generate_type_list(types.Grid, types.RHS, types.Finished))
    pset.addTerminal((approximation, rhs), multiple.generate_type_list(types.Grid, types.RHS, types.NotFinished), 'u_and_f')
    pset.addTerminal(terminals.no_partitioning, types.Partitioning, f'no')
    pset.addTerminal(terminals.red_black_partitioning, types.Partitioning, f'red_black')
    # pset.addPrimitive(lambda x: x + 1, [int], int, 'inc')

    add_cycle(pset, terminals, types, 0, coarsest)
    for i in range(1, depth):
        approximation = system.ZeroApproximation(terminals.coarse_grid)
        operator = coarse_operator
        prolongation = coarse_prolongation
        restriction = coarse_restriction
        fine_grid = terminals.coarse_grid
        coarse_grid = system.get_coarse_grid(fine_grid, coarsening_factors)
        cgs_expression = None
        coarsest = False
        if i == depth - 1:
            coarsest = True
            cgs_expression = coarse_grid_solver_expression
            coarse_operator = \
                generate_system_operator_from_l2_information(equations, operators, fields, max_level - i - 1,
                                                             coarse_grid)
        else:
            coarse_operator, coarse_restriction, coarse_prolongation = \
                generate_operators_from_l2_information(equations, operators, fields, max_level - i - 1, coarse_grid,
                                                       system.get_coarse_grid(coarse_grid, coarsening_factors))

        terminals = Terminals(approximation, dimension, coarsening_factors, operator, coarse_operator, restriction,
                              prolongation, cgs_expression)
        types = Types(terminals, LevelFinishedType, LevelNotFinishedType)
        add_cycle(pset, terminals, types, i, coarsest)

    return pset