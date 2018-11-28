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

        self.diagonal = base.Diagonal(operator)
        self.block_diagonal = base.BlockDiagonal(operator, tuple(2 for _ in range(self.dimension)))
        self.lower = base.LowerTriangle(operator)
        self.upper = base.UpperTriangle(operator)
        self.coarse_grid = mg.get_coarse_grid(self.grid, self.coarsening_factor)
        self.coarse_operator = mg.get_coarse_operator(self.operator, self.coarse_grid)
        self.identity = base.Identity(self.operator.shape, self.grid)
        self.coarse_grid_solver = mg.CoarseGridSolver(self.coarse_operator)
        self.no_partitioning = part.Single
        self.red_black_partitioning = part.RedBlack


class Types:
    def __init__(self, terminals: Terminals, Finished, NotFinished):
        self.Operator = matrix_types.generate_matrix_type(terminals.operator.shape)
        self.LowerTriangularOperator = matrix_types.generate_lower_triangular_matrix_type(terminals.lower.shape)
        self.UpperTriangularOperator = matrix_types.generate_upper_triangular_matrix_type(terminals.lower.shape)
        self.Grid = grid_types.generate_grid_type(terminals.grid.size)
        self.Correction = grid_types.generate_correction_type(terminals.grid.size)
        self.RHS = grid_types.generate_rhs_type(terminals.grid.size)
        self.DiagonalOperator = matrix_types.generate_diagonal_matrix_type(terminals.diagonal.shape)
        self.BlockDiagonalOperator = matrix_types.generate_block_diagonal_matrix_type(terminals.block_diagonal.shape)
        self.Interpolation = matrix_types.generate_matrix_type(terminals.interpolation.shape)
        self.Restriction = matrix_types.generate_matrix_type(terminals.restriction.shape)
        self.CoarseOperator = matrix_types.generate_matrix_type(terminals.coarse_operator.shape)
        self.CoarseGridSolver = matrix_types.generate_solver_type(terminals.coarse_operator.shape)
        self.CoarseGrid = grid_types.generate_grid_type(terminals.coarse_grid.size)
        self.CoarseRHS = grid_types.generate_rhs_type(terminals.coarse_grid.size)
        self.CoarseCorrection = grid_types.generate_correction_type(terminals.coarse_grid.size)
        self.Partitioning = part.Partitioning
        self.Finished = Finished
        self.NotFinished = NotFinished


def add_cycle(pset: gp.PrimitiveSetTyped, terminals: Terminals, types, level, coarsest=False):
    # pset.addTerminal(terminals.grid, types.Grid, f'u_{level}')
    null_grid_coarse = base.ZeroGrid(terminals.coarse_grid.size, terminals.coarse_grid.step_size)
    pset.addTerminal(null_grid_coarse, types.CoarseGrid, f'zero_grid_{level+1}')
    # pset.addTerminal(terminals.operator, types.Operator, f'A_{level}')
    # pset.addTerminal(terminals.identity, types.DiagonalOperator, f'I_{level}')
    # pset.addTerminal(terminals.diagonal, types.DiagonalOperator, f'D_{level}')
    pset.addTerminal(base.inv(terminals.diagonal), types.DiagonalOperator, f'D_inv_{level}')
    # pset.addTerminal(terminals.lower, types.LowerTriangularOperator, f'L_{level}')
    # pset.addTerminal(terminals.upper, types.UpperTriangularOperator, f'U_{level}')
    # pset.addTerminal(terminals.block_diagonal, types.BlockDiagonalOperator, f'BD_{level}')
    # pset.addTerminal(terminals.coarse_grid_solver, types.CoarseGridSolver, f'S_{level}')
    pset.addTerminal(terminals.interpolation, types.Interpolation, f'P_{level}')
    pset.addTerminal(terminals.restriction, types.Restriction, f'R_{level}')

    OperatorType = types.Operator
    GridType = types.Grid
    RHSType = types.RHS
    CorrectionType = types.Correction
    DiagonalOperatorType = types.DiagonalOperator
    # BlockDiagonalOperatorType = types.BlockDiagonalOperator
    # pset.addPrimitive(base.add, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, f'add_{level}')
    # pset.addPrimitive(base.add, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'add_{level}')
    # pset.addPrimitive(base.add, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'add_{level}')
    # pset.addPrimitive(base.add, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, f'add_{level}')
    # pset.addPrimitive(base.add, [OperatorType, OperatorType], OperatorType, f'add_{level}')

    # pset.addPrimitive(base.sub, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, f'sub_{level}')
    # pset.addPrimitive(base.sub, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'sub_{level}')
    # pset.addPrimitive(base.sub, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'sub_{level}')
    # pset.addPrimitive(base.sub, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, f'sub_{level}')
    # pset.addPrimitive(base.sub, [OperatorType, OperatorType], OperatorType, f'sub_{level}')

    # pset.addPrimitive(base.mul, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, f'mul_{level}')
    # pset.addPrimitive(base.mul, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'mul_{level}')
    # pset.addPrimitive(base.mul, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, f'mul_{level}')
    # pset.addPrimitive(base.mul, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, f'mul_{level}')
    # pset.addPrimitive(base.mul, [OperatorType, OperatorType], OperatorType, f'mul_{level}')
    # pset.addPrimitive(base.minus, [OperatorType], OperatorType, f'minus_{level}')

    # pset.addPrimitive(base.inv, [DiagonalOperatorType], DiagonalOperatorType, f'inverse_{level}')
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

    def reduce(cycle, times):
        from evostencils.expressions import transformations
        new_cycle = transformations.repeat(cycle, times)
        return new_cycle, new_cycle.rhs



    pset.addPrimitive(reduce, [multiple.generate_type_list(types.Grid, types.Correction, types.Finished), int], multiple.generate_type_list(types.Grid, types.RHS, types.Finished), f"reduce_{level}")
    pset.addPrimitive(reduce, [multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished), int], multiple.generate_type_list(types.Grid, types.RHS, types.NotFinished), f"reduce_{level}")

    pset.addPrimitive(create_cycle_on_current_level, [multiple.generate_type_list(types.Grid, types.RHS, types.Finished), part.Partitioning], multiple.generate_type_list(GridType, CorrectionType, types.Finished), f"cycle_on_level_{level}")
    pset.addPrimitive(create_cycle_on_current_level, [multiple.generate_type_list(types.Grid, types.RHS, types.NotFinished), part.Partitioning], multiple.generate_type_list(GridType, CorrectionType, types.NotFinished), f"cycle_on_level_{level}")

    pset.addPrimitive(extend, [OperatorType, multiple.generate_type_list(types.Grid, types.Correction, types.Finished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.Finished),
                      f"extend_{level}")
    pset.addPrimitive(extend, [OperatorType, multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished),
                      f"extend_{level}")
    if not coarsest:

        pset.addPrimitive(move_level_up, [multiple.generate_type_list(types.CoarseGrid, types.CoarseCorrection, types.Finished)], multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished), f"move_up_{level}")

        pset.addPrimitive(create_cycle_on_lower_level,
                          [types.CoarseGrid, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.NotFinished),
                           part.Partitioning],
                          multiple.generate_type_list(types.CoarseGrid, types.CoarseCorrection, types.NotFinished),
                          f"new_cycle_on_lower_level_{level}")
        pset.addPrimitive(create_cycle_on_lower_level,
                          [types.CoarseGrid, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished),
                           part.Partitioning],
                          multiple.generate_type_list(types.CoarseGrid, types.CoarseCorrection, types.Finished),
                          f"new_cycle_on_lower_level_{level}")

    else:
        pset.addTerminal(terminals.coarse_grid_solver, types.CoarseGridSolver, f'S_{level}')
        pset.addPrimitive(extend, [types.CoarseGridSolver, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.NotFinished)],
                          multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished),
                          f'solve_{level}')
        pset.addPrimitive(extend, [types.CoarseGridSolver, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished)],
                          multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished),
                          f'solve_{level}')

    # Multigrid recipes
    pset.addPrimitive(extend, [types.Restriction, multiple.generate_type_list(types.Grid, types.Correction, types.Finished)],
                      multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished),
                      f'restrict_{level}')
    pset.addPrimitive(extend, [types.Restriction, multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished)],
                      multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.NotFinished),
                      f'restrict_{level}')

    pset.addPrimitive(extend, [types.Interpolation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.Finished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.Finished),
                      f'interpolate_{level}')
    pset.addPrimitive(extend, [types.Interpolation, multiple.generate_type_list(types.Grid, types.CoarseCorrection, types.NotFinished)],
                      multiple.generate_type_list(types.Grid, types.Correction, types.NotFinished),
                      f'interpolate_{level}')


def generate_primitive_set(operator, grid, rhs, dimension, coarsening_factor,
                           interpolation, restriction, depth=4):
    assert depth >= 1, "The maximum number of cycles must be greater zero"
    terminals = Terminals(operator, grid, dimension, coarsening_factor, interpolation, restriction)
    Finished = multiple.generate_new_type('Finished')
    NotFinished = multiple.generate_new_type('NotFinished')
    types = Types(terminals, Finished, NotFinished)
    pset = gp.PrimitiveSetTyped("main", [], multiple.generate_type_list(types.Grid, types.RHS, types.Finished))
    pset.addTerminal((grid, rhs), multiple.generate_type_list(types.Grid, types.RHS, types.NotFinished), 'u_and_f')
    pset.addTerminal(terminals.no_partitioning, types.Partitioning, f'no')
    pset.addTerminal(terminals.red_black_partitioning, types.Partitioning, f'red_black')
    pset.addTerminal(1, int)
    #pset.addTerminal(2, int)
    #pset.addTerminal(3, int)
    # TODO more steps are probably not a good idea as the LFA seems to crash often
    # pset.addTerminal(4, int)
    # pset.addTerminal(5, int)
    #pset.addPrimitive(lambda x: x + 1, [int], int, 'inc')

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
        types = Types(terminals, Finished, NotFinished)
        add_cycle(pset, terminals, types, i, coarsest)

    return pset
