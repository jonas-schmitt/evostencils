from evostencils.expressions import multigrid as mg
from evostencils.expressions import base
from evostencils.expressions import partitioning as part
from evostencils.types import matrix as matrix_types
from evostencils.types import grid as grid_types
from evostencils.types import multiple
from deap import gp


class Terminals:
    def __init__(self, operator, grid, dimension, coarsening_factor, interpolation, restriction):
        self.operator = operator
        self.grid = grid
        self.dimension = dimension
        self.coarsening_factor = coarsening_factor
        self.interpolation = interpolation
        self.restriction = restriction
        self.coarse_grid = mg.get_coarse_grid(self.grid, self.coarsening_factor)
        self.coarse_operator = mg.get_coarse_operator(self.operator, self.coarse_grid)
        self.identity = base.Identity(self.operator.shape, self.grid)
        self.coarse_grid_solver = mg.CoarseGridSolver(self.coarse_operator)
        self.no_partitioning = part.Single
        self.red_black_partitioning = part.RedBlack


class Types:
    def __init__(self, terminals: Terminals, LevelFinishedType, LevelNotFinishedType):
        self.Operator = matrix_types.generate_matrix_type(terminals.operator.shape)
        self.LowerTriangularOperator = matrix_types.generate_lower_triangular_matrix_type(terminals.operator.shape)
        self.UpperTriangularOperator = matrix_types.generate_upper_triangular_matrix_type(terminals.operator.shape)
        self.Grid = grid_types.generate_grid_type(terminals.grid.size)
        self.Correction = grid_types.generate_correction_type(terminals.grid.size)
        self.RHS = grid_types.generate_rhs_type(terminals.grid.size)
        self.DiagonalOperator = matrix_types.generate_diagonal_matrix_type(terminals.operator.shape)
        self.BlockDiagonalOperator = matrix_types.generate_block_diagonal_matrix_type(terminals.operator.shape)
        self.Interpolation = matrix_types.generate_matrix_type(terminals.interpolation.shape)
        self.Restriction = matrix_types.generate_matrix_type(terminals.restriction.shape)
        self.CoarseOperator = matrix_types.generate_matrix_type(terminals.coarse_operator.shape)
        self.CoarseGridSolver = matrix_types.generate_solver_type(terminals.coarse_operator.shape)
        self.CoarseGrid = grid_types.generate_grid_type(terminals.coarse_grid.size)
        self.CoarseRHS = grid_types.generate_rhs_type(terminals.coarse_grid.size)
        self.CoarseCorrection = grid_types.generate_correction_type(terminals.coarse_grid.size)
        self.Partitioning = part.Partitioning
        self.LevelFinished = LevelFinishedType
        self.LevelNotFinished = LevelNotFinishedType


def add_cycle(pset: gp.PrimitiveSetTyped, terminals: Terminals, types: Types, level, coarsest=False):
    # pset.addTerminal(terminals.grid, types.Grid, f'u_{level}')
    null_grid_coarse = base.ZeroGrid(terminals.coarse_grid.size, terminals.coarse_grid.step_size)
    pset.addTerminal(null_grid_coarse, types.CoarseGrid, f'zero_grid_{level+1}')
    pset.addTerminal(terminals.operator, types.Operator, f'A_{level}')
    pset.addTerminal(terminals.identity, types.DiagonalOperator, f'I_{level}')
    pset.addTerminal(base.Diagonal(terminals.operator), types.DiagonalOperator, f'D_{level}')
    pset.addTerminal(base.inv(base.Diagonal(terminals.operator)), types.DiagonalOperator, f'D_inv_{level}')
    # pset.addTerminal(terminals.block_diagonal_2, types.BlockDiagonalOperator, f'BD2_{level}')
    # pset.addTerminal(terminals.block_diagonal_3, types.BlockDiagonalOperator, f'BD3_{level}')
    # pset.addTerminal(terminals.block_diagonal_4, types.BlockDiagonalOperator, f'BD4_{level}')
    pset.addTerminal(terminals.interpolation, types.Interpolation, f'P_{level}')
    pset.addTerminal(terminals.restriction, types.Restriction, f'R_{level}')

    OperatorType = types.Operator
    GridType = types.Grid
    RHSType = types.RHS
    CorrectionType = types.Correction
    DiagonalOperatorType = types.DiagonalOperator
    # BlockDiagonalOperatorType = types.BlockDiagonalOperator
    pset.addPrimitive(base.add, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, f'add_{level}')
    # pset.addPrimitive(base.add, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'add_{level}')
    # pset.addPrimitive(base.add, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'add_{level}')
    # pset.addPrimitive(base.add, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, f'add_{level}')
    pset.addPrimitive(base.add, [OperatorType, OperatorType], OperatorType, f'add_{level}')

    pset.addPrimitive(base.sub, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, f'sub_{level}')
    # pset.addPrimitive(base.sub, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'sub_{level}')
    # pset.addPrimitive(base.sub, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'sub_{level}')
    # pset.addPrimitive(base.sub, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, f'sub_{level}')
    pset.addPrimitive(base.sub, [OperatorType, OperatorType], OperatorType, f'sub_{level}')

    pset.addPrimitive(base.mul, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, f'mul_{level}')
    # pset.addPrimitive(base.mul, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'mul_{level}')
    # pset.addPrimitive(base.mul, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'mul_{level}')
    # pset.addPrimitive(base.mul, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, f'mul_{level}')
    pset.addPrimitive(base.mul, [OperatorType, OperatorType], OperatorType, f'mul_{level}')
    pset.addPrimitive(base.minus, [OperatorType], OperatorType, f'minus_{level}')

    # pset.addPrimitive(base.diag, [OperatorType], DiagonalOperatorType, f'diag_{level}')
    pset.addPrimitive(base.inv, [DiagonalOperatorType], DiagonalOperatorType, f'inverse_{level}')
    # pset.addPrimitive(base.inv, [BlockDiagonalOperatorType], OperatorType, f'inverse_{level}')

    def create_cycle_on_lower_level(coarse_grid, cycle, partitioning):
        result = mg.cycle(cycle.iterate, cycle.rhs,
                          mg.cycle(coarse_grid, cycle.correction, mg.residual(terminals.coarse_operator, coarse_grid, cycle.correction),
                                   partitioning), cycle.partitioning, predecessor=cycle.predecessor)
        result.correction.predecessor = result
        return result.correction

    def create_cycle_on_current_level(args, partitioning):
        return mg.cycle(args[0], args[1], mg.residual(terminals.operator, args[0], args[1]), partitioning, predecessor=args[0].predecessor)

    def extend(operator, cycle):
        return mg.cycle(cycle.iterate, cycle.rhs, base.mul(operator, cycle.correction), cycle.partitioning, cycle.weight,
                        cycle.predecessor)

    def move_level_up(cycle):
        cycle.predecessor._correction = cycle
        return cycle.predecessor

    def reduce(cycle, times=1):
        from evostencils.expressions import transformations
        new_cycle = transformations.repeat(cycle, times)
        return new_cycle, new_cycle.rhs

    pset.addPrimitive(reduce, [multiple.generate_type_list(types.Grid, types.Correction, types.LevelFinished), int], multiple.generate_type_list(types.Grid, types.RHS, types.LevelFinished), f"reduce_{level}")
    pset.addPrimitive(reduce, [multiple.generate_type_list(types.Grid, types.Correction, types.LevelNotFinished), int], multiple.generate_type_list(types.Grid, types.RHS, types.LevelNotFinished), f"reduce_{level}")

    pset.addPrimitive(create_cycle_on_current_level, [multiple.generate_type_list(types.Grid, types.RHS, types.LevelFinished), part.Partitioning], multiple.generate_type_list(GridType, CorrectionType, types.LevelFinished), f"cycle_on_level_{level}")
    pset.addPrimitive(create_cycle_on_current_level, [multiple.generate_type_list(types.Grid, types.RHS, types.LevelNotFinished), part.Partitioning], multiple.generate_type_list(GridType, CorrectionType, types.LevelNotFinished), f"cycle_on_level_{level}")
    pset.addPrimitive(extend, [OperatorType, multiple.generate_type_list(types.Grid, types.Correction, types.LevelFinished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.LevelFinished),
                      f"extend_{level}")
    pset.addPrimitive(extend, [OperatorType, multiple.generate_type_list(types.Grid, types.Correction, types.LevelNotFinished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.LevelNotFinished),
                      f"extend_{level}")

    """

    def smooth(args, operator, partitioning, times):
        cycle = create_cycle_on_current_level(args, partitioning)
        cycle = extend(operator, cycle)
        return reduce(cycle, times)

    pset.addPrimitive(smooth, [multiple.generate_type_list(types.Grid, types.RHS, types.LevelFinished), OperatorType, part.Partitioning, int], multiple.generate_type_list(types.Grid, types.RHS, types.LevelFinished), f'smooth_{level}')
    pset.addPrimitive(smooth, [multiple.generate_type_list(types.Grid, types.RHS, types.LevelNotFinished), OperatorType, part.Partitioning, int], multiple.generate_type_list(types.Grid, types.RHS, types.LevelNotFinished), f'smooth_{level}')
    """

    if not coarsest:

        pset.addPrimitive(move_level_up, [multiple.generate_type_list(types.CoarseGrid, types.CoarseCorrection, types.LevelFinished)], multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelFinished), f"move_up_{level}")

        pset.addPrimitive(create_cycle_on_lower_level,
                          [types.CoarseGrid, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelNotFinished),
                           part.Partitioning],
                          multiple.generate_type_list(types.CoarseGrid, types.CoarseCorrection, types.LevelNotFinished),
                          f"new_cycle_on_lower_level_{level}")
        pset.addPrimitive(create_cycle_on_lower_level,
                          [types.CoarseGrid, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelFinished),
                           part.Partitioning],
                          multiple.generate_type_list(types.CoarseGrid, types.CoarseCorrection, types.LevelFinished),
                          f"new_cycle_on_lower_level_{level}")

    else:
        pset.addTerminal(terminals.coarse_grid_solver, types.CoarseGridSolver, f'S_{level}')
        pset.addPrimitive(extend, [types.CoarseGridSolver, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelNotFinished)],
                          multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelFinished),
                          f'solve_{level}')
        pset.addPrimitive(extend, [types.CoarseGridSolver, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelFinished)],
                          multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelFinished),
                          f'solve_{level}')

    # Multigrid recipes
    pset.addPrimitive(extend, [types.Restriction, multiple.generate_type_list(types.Grid, types.Correction, types.LevelFinished)],
                      multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelFinished),
                      f'restrict_{level}')
    pset.addPrimitive(extend, [types.Restriction, multiple.generate_type_list(types.Grid, types.Correction, types.LevelNotFinished)],
                      multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelNotFinished),
                      f'restrict_{level}')

    pset.addPrimitive(extend, [types.Interpolation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelFinished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.LevelFinished),
                      f'interpolate_{level}')
    pset.addPrimitive(extend, [types.Interpolation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.LevelNotFinished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.LevelNotFinished),
                      f'interpolate_{level}')


def generate_primitive_set(operator, grid, rhs, dimension, coarsening_factor,
                           interpolation, restriction, depth=4, LevelFinishedType=None, LevelNotFinishedType=None):
    assert depth >= 1, "The maximum number of cycles must be greater zero"
    terminals = Terminals(operator, grid, dimension, coarsening_factor, interpolation, restriction)
    if LevelFinishedType is None:
        LevelFinishedType = multiple.generate_new_type('LevelFinished')
    if LevelNotFinishedType is None:
        LevelNotFinishedType = multiple.generate_new_type('LevelNotFinished')
    types = Types(terminals, LevelFinishedType, LevelNotFinishedType)
    pset = gp.PrimitiveSetTyped("main", [], multiple.generate_type_list(types.Grid, types.RHS, types.LevelFinished))
    pset.addTerminal((grid, rhs), multiple.generate_type_list(types.Grid, types.RHS, types.LevelNotFinished), 'u_and_f')
    pset.addTerminal(terminals.no_partitioning, types.Partitioning, f'no')
    pset.addTerminal(terminals.red_black_partitioning, types.Partitioning, f'red_black')
    pset.addTerminal(1, int)
    # pset.addTerminal(2, int)
    # pset.addTerminal(3, int)
    # pset.addTerminal(4, int)
    # pset.addTerminal(5, int)
    pset.addPrimitive(lambda x: x + 1, [int], int, 'inc')

    coarsest = False
    if depth == 1:
        coarsest = True
    add_cycle(pset, terminals, types, 0, coarsest)
    for i in range(1, depth):
        coarse_grid = base.ZeroGrid(terminals.coarse_grid.size, terminals.coarse_grid.step_size)
        coarse_interpolation = mg.get_interpolation(coarse_grid, mg.get_coarse_grid(coarse_grid, coarsening_factor), interpolation.stencil_generator)
        coarse_restriction = mg.get_restriction(coarse_grid, mg.get_coarse_grid(coarse_grid, coarsening_factor), restriction.stencil_generator)
        terminals = Terminals(terminals.coarse_operator, coarse_grid, dimension, coarsening_factor,
                              coarse_interpolation, coarse_restriction)
        coarsest = False
        if i == depth - 1:
            coarsest = True
        types = Types(terminals, LevelFinishedType, LevelNotFinishedType)
        add_cycle(pset, terminals, types, i, coarsest)

    return pset
