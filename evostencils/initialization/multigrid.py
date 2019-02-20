from evostencils.expressions import multigrid as mg
from evostencils.expressions import base
from evostencils.expressions import partitioning as part
from evostencils.types import operator as matrix_types
from evostencils.types import grid as grid_types
from evostencils.types import multiple
from evostencils.types import partitioning, level_control
from evostencils.optimization.deap_extension import PrimitiveSetTyped
from deap import gp


class Terminals:
    def __init__(self, operator, grid, dimension, coarsening_factor, interpolation, restriction,
                 coarse_grid_solver_expression=None):
        self.operator = operator
        self.grid = grid
        self.dimension = dimension
        self.coarsening_factor = coarsening_factor
        self.interpolation = interpolation
        self.restriction = restriction
        self.coarse_grid = mg.get_coarse_grid(self.grid, self.coarsening_factor)
        self.coarse_operator = mg.get_coarse_operator(self.operator, self.coarse_grid)
        self.identity = base.Identity(self.operator.shape, self.grid)
        self.coarse_grid_solver = mg.CoarseGridSolver(self.coarse_operator, expression=coarse_grid_solver_expression)
        self.no_partitioning = part.Single
        self.red_black_partitioning = part.RedBlack


class Types:
    def __init__(self, terminals: Terminals, FinishedType, NotFinishedType):
        self.Operator = matrix_types.generate_operator_type(terminals.operator.shape)
        self.LowerTriangularOperator = matrix_types.generate_lower_triangular_operator_type(terminals.operator.shape)
        self.UpperTriangularOperator = matrix_types.generate_upper_triangular_operator_type(terminals.operator.shape)
        self.Grid = grid_types.generate_grid_type(terminals.grid.size)
        self.Correction = grid_types.generate_correction_type(terminals.grid.size)
        self.RHS = grid_types.generate_rhs_type(terminals.grid.size)
        self.DiagonalOperator = matrix_types.generate_diagonal_operator_type(terminals.operator.shape)
        self.BlockDiagonalOperator = matrix_types.generate_block_diagonal_operator_type(terminals.operator.shape)
        self.Interpolation = matrix_types.generate_operator_type(terminals.interpolation.shape)
        self.Restriction = matrix_types.generate_operator_type(terminals.restriction.shape)
        self.CoarseOperator = matrix_types.generate_operator_type(terminals.coarse_operator.shape)
        self.CoarseGridSolver = matrix_types.generate_solver_type(terminals.coarse_operator.shape)
        self.CoarseGrid = grid_types.generate_grid_type(terminals.coarse_grid.size)
        self.CoarseRHS = grid_types.generate_rhs_type(terminals.coarse_grid.size)
        self.CoarseCorrection = grid_types.generate_correction_type(terminals.coarse_grid.size)
        self.Partitioning = partitioning.generate_any_partitioning_type()
        self.Finished = FinishedType
        self.NotFinished = NotFinishedType


def add_cycle(pset: gp.PrimitiveSetTyped, terminals: Terminals, types: Types, level, coarsest=False):
    null_grid_coarse = base.ZeroGrid(terminals.coarse_grid.size, terminals.coarse_grid.step_size)
    pset.addTerminal(null_grid_coarse, types.CoarseGrid, f'zero_grid_{level+1}')
    # pset.addTerminal(terminals.operator, types.Operator, f'A_{level}')
    # pset.addTerminal(terminals.identity, types.DiagonalOperator, f'I_{level}')
    # pset.addTerminal(base.Diagonal(terminals.operator), types.DiagonalOperator, f'D_{level}')
    # pset.addTerminal(base.LowerTriangle(terminals.operator), types.Operator, f'L_{level}')
    # pset.addTerminal(base.UpperTriangle(terminals.operator), types.Operator, f'U_{level}')
    pset.addTerminal(base.inv(base.Diagonal(terminals.operator)), types.DiagonalOperator, f'D_inv_{level}')
    pset.addTerminal(terminals.interpolation, types.Interpolation, f'P_{level}')
    pset.addTerminal(terminals.restriction, types.Restriction, f'R_{level}')

    OperatorType = types.Operator
    GridType = types.Grid
    CorrectionType = types.Correction
    DiagonalOperatorType = types.DiagonalOperator
    # pset.addPrimitive(base.add, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, f'add_{level}')
    # pset.addPrimitive(base.add, [OperatorType, OperatorType], OperatorType, f'add_{level}')

    # pset.addPrimitive(base.sub, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, f'sub_{level}')
    # pset.addPrimitive(base.sub, [OperatorType, OperatorType], OperatorType, f'sub_{level}')

    # pset.addPrimitive(base.mul, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, f'mul_{level}')
    # pset.addPrimitive(base.mul, [OperatorType, OperatorType], OperatorType, f'mul_{level}')

    # pset.addPrimitive(base.inv, [DiagonalOperatorType], DiagonalOperatorType, f'inverse_{level}')
    # pset.addPrimitive(base.minus, [OperatorType], OperatorType, f'minus_{level}')

    # BlockDiagonalOperatorType = types.BlockDiagonalOperator
    # pset.addTerminal(terminals.block_diagonal_2, types.BlockDiagonalOperator, f'BD2_{level}')
    # pset.addTerminal(terminals.block_diagonal_3, types.BlockDiagonalOperator, f'BD3_{level}')
    # pset.addTerminal(terminals.block_diagonal_4, types.BlockDiagonalOperator, f'BD4_{level}')
    # pset.addPrimitive(base.add, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'add_{level}')
    # pset.addPrimitive(base.add, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'add_{level}')
    # pset.addPrimitive(base.add, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, f'add_{level}')
    # pset.addPrimitive(base.sub, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'sub_{level}')
    # pset.addPrimitive(base.sub, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'sub_{level}')
    # pset.addPrimitive(base.sub, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, f'sub_{level}')
    # pset.addPrimitive(base.mul, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'mul_{level}')
    # pset.addPrimitive(base.mul, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'mul_{level}')
    # pset.addPrimitive(base.mul, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, f'mul_{level}')
    # pset.addPrimitive(base.inv, [BlockDiagonalOperatorType], OperatorType, f'inverse_{level}')

    def coarse_cycle(coarse_grid, cycle, partitioning):
        result = mg.cycle(cycle.iterate, cycle.rhs,
                          mg.cycle(coarse_grid, cycle.correction, mg.residual(terminals.coarse_operator, coarse_grid, cycle.correction),
                                   partitioning), cycle.partitioning, predecessor=cycle.predecessor)
        result.correction.predecessor = result
        return result.correction

    def residual(args, partitioning):
        return mg.cycle(args[0], args[1], mg.residual(terminals.operator, args[0], args[1]), partitioning, predecessor=args[0].predecessor)

    def apply(operator, cycle):
        return mg.cycle(cycle.iterate, cycle.rhs, base.mul(operator, cycle.correction), cycle.partitioning, cycle.weight,
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

        pset.addPrimitive(coarse_grid_correction, [types.Interpolation, multiple.generate_type_list(types.CoarseGrid, types.CoarseCorrection, types.Finished)], multiple.generate_type_list(types.Grid, types.RHS, types.Finished), f"cgc_{level}")

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
            new_cycle = mg.cycle(cycle.iterate, cycle.rhs, base.mul(interpolation, base.mul(cgs, cycle.correction)), cycle.partitioning, cycle.weight,
                        cycle.predecessor)
            return iterate(new_cycle)

        pset.addPrimitive(solve, [types.CoarseGridSolver, types.Interpolation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.NotFinished)],
                          multiple.generate_type_list(types.Grid, types.RHS, types.Finished),
                          f'solve_{level}')
        pset.addPrimitive(solve, [types.CoarseGridSolver, types.Interpolation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished)],
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


def generate_primitive_set(operator, grid, rhs, dimension, coarsening_factor,
                           interpolation, restriction, coarse_grid_solver_expression=None, depth=2, LevelFinishedType=None, LevelNotFinishedType=None):
    assert depth >= 1, "The maximum number of cycles must be greater zero"
    coarsest = False
    cgs_expression = None
    if depth == 1:
        coarsest = True
        cgs_expression = coarse_grid_solver_expression
    terminals = Terminals(operator, grid, dimension, coarsening_factor, interpolation, restriction, cgs_expression)
    if LevelFinishedType is None:
        LevelFinishedType = level_control.generate_finished_type()
    if LevelNotFinishedType is None:
        LevelNotFinishedType = level_control.generate_not_finished_type()
    types = Types(terminals, LevelFinishedType, LevelNotFinishedType)
    pset = PrimitiveSetTyped("main", [], multiple.generate_type_list(types.Grid, types.RHS, types.Finished))
    pset.addTerminal((grid, rhs), multiple.generate_type_list(types.Grid, types.RHS, types.NotFinished), 'u_and_f')
    pset.addTerminal(terminals.no_partitioning, types.Partitioning, f'no')
    pset.addTerminal(terminals.red_black_partitioning, types.Partitioning, f'red_black')
    # pset.addPrimitive(lambda x: x + 1, [int], int, 'inc')

    add_cycle(pset, terminals, types, 0, coarsest)
    for i in range(1, depth):
        coarse_grid = base.ZeroGrid(terminals.coarse_grid.size, terminals.coarse_grid.step_size)
        coarse_interpolation = mg.get_interpolation(coarse_grid, mg.get_coarse_grid(coarse_grid, coarsening_factor), interpolation.stencil_generator)
        coarse_restriction = mg.get_restriction(coarse_grid, mg.get_coarse_grid(coarse_grid, coarsening_factor), restriction.stencil_generator)
        cgs_expression = None
        coarsest = False
        if i == depth - 1:
            coarsest = True
            cgs_expression = coarse_grid_solver_expression

        terminals = Terminals(terminals.coarse_operator, coarse_grid, dimension, coarsening_factor,
                              coarse_interpolation, coarse_restriction, cgs_expression)
        types = Types(terminals, LevelFinishedType, LevelNotFinishedType)
        add_cycle(pset, terminals, types, i, coarsest)

    return pset
