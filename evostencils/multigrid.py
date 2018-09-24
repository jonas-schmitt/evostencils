from evostencils.expressions import multigrid as mg
from evostencils.expressions import base
from evostencils.expressions import partitioning as part
from evostencils.stencils import constant
from evostencils import matrix_types
from deap import gp


class Terminals:
    def __init__(self, operator, grid, dimension, coarsening_factor, interpolation_stencil=None, restriction_stencil=None):
        self.operator = operator
        self.grid = grid
        self.dimension = dimension
        self.coarsening_factor = coarsening_factor
        if interpolation_stencil is None:
            interpolation_stencil_entries = [
                ((-1, -1), 1.0/4),
                (( 0, -1), 1.0/2),
                (( 1, -1), 1.0/4),
                ((-1,  0), 1.0/2),
                (( 0,  0), 1.0),
                (( 1,  0), 1.0/2),
                ((-1,  1), 1.0/4),
                (( 0,  1), 1.0/2),
                (( 1,  1), 1.0/4),
            ]
            self.interpolation_stencil = constant.Stencil(interpolation_stencil_entries)
        else:
            self.interpolation_stencil = interpolation_stencil

        if restriction_stencil is None:
            restriction_stencil_entries = [
                ((-1, -1), 1.0/16),
                (( 0, -1), 1.0/8),
                (( 1, -1), 1.0/16),
                ((-1,  0), 1.0/8),
                (( 0,  0), 1.0/4),
                (( 1,  0), 1.0/8),
                ((-1,  1), 1.0/16),
                (( 0,  1), 1.0/8),
                (( 1,  1), 1.0/16),
            ]
            self.restriction_stencil = constant.Stencil(restriction_stencil_entries)
        else:
            self.restriction_stencil = restriction_stencil

        self.diagonal = base.Diagonal(operator)
        self.block_diagonal = base.BlockDiagonal(operator, tuple(2 for _ in range(self.dimension)))
        self.lower = base.LowerTriangle(operator)
        self.upper = base.UpperTriangle(operator)
        self.coarse_grid = mg.get_coarse_grid(self.grid, self.coarsening_factor)
        self.coarse_operator = mg.get_coarse_operator(self.operator, self.coarsening_factor)
        self.interpolation = mg.get_interpolation(self.grid, self.coarse_grid, self.interpolation_stencil)
        self.restriction = mg.get_interpolation(self.grid, self.coarse_grid, self.interpolation_stencil)
        self.identity = base.Identity(self.operator.shape, self.dimension)
        self.coarse_grid_solver = mg.CoarseGridSolver(self.coarse_grid)
        self.no_partitioning = part.Single
        self.red_black_partitioning = part.RedBlack


class Types:
    def __init__(self, terminals: Terminals):
        self.Operator = matrix_types.generate_matrix_type(terminals.operator.shape)
        self.LowerTriangularOperator = matrix_types.generate_lower_triangular_matrix_type(terminals.lower.shape)
        self.UpperTriangularOperator = matrix_types.generate_upper_triangular_matrix_type(terminals.lower.shape)
        self.Grid = matrix_types.generate_matrix_type(terminals.grid.shape)
        self.DiagonalOperator = matrix_types.generate_diagonal_matrix_type(terminals.diagonal.shape)
        self.BlockDiagonalOperator = matrix_types.generate_block_diagonal_matrix_type(terminals.block_diagonal)
        self.Interpolation = matrix_types.generate_matrix_type(terminals.interpolation.shape)
        self.Restriction = matrix_types.generate_matrix_type(terminals.restriction.shape)
        self.CoarseOperator = matrix_types.generate_matrix_type(terminals.coarse_operator.shape)
        self.CoarseGrid = matrix_types.generate_matrix_type(terminals.coarse_grid.shape)
        self.Partitioning = part.Partitioning


def add_multigrid_cycle(pset: gp.PrimitiveSetTyped, terminals: Terminals, types=None):
    if types is None:
        types = Types(terminals)
    pset.addTerminal(terminals.operator, types.Operator, 'A')
    pset.addTerminal(terminals.identity, types.DiagonalOperator, 'I')
    pset.addTerminal(terminals.diagonal, types.DiagonalOperator, 'D')
    pset.addTerminal(terminals.lower, types.LowerTriangularOperator, 'L')
    pset.addTerminal(terminals.upper, types.UpperTriangularOperator, 'U')
    pset.addTerminal(terminals.block_diagonal, types.BlockDiagonalOperator, 'BD')
    pset.addTerminal(terminals.coarse_grid_solver, types.CoarseOperator, 'S')
    # Note: Omitted zero matrix, because it should not be required
    pset.addTerminal(terminals.interpolation, types.Interpolation, 'P')
    pset.addTerminal(terminals.restriction, types.Restriction, 'R')
    pset.addTerminal(terminals.no_partitioning, types.Partitioning, 'no_partitioning')
    pset.addTerminal(terminals.red_black_partitioning, types.Partitioning, 'red_black')

    OperatorType = types.Operator
    GridType = types.Grid
    DiagonalOperatorType = types.DiagonalOperator
    BlockDiagonalOperatorType = types.BlockDiagonalOperator

    pset.addPrimitive(base.add, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, 'add')
    pset.addPrimitive(base.add, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, 'add')
    pset.addPrimitive(base.add, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, 'add')
    pset.addPrimitive(base.add, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, 'add')
    pset.addPrimitive(base.add, [OperatorType, OperatorType], OperatorType, 'add')

    pset.addPrimitive(base.sub, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, 'sub')
    pset.addPrimitive(base.sub, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, 'sub')
    pset.addPrimitive(base.sub, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, 'sub')
    pset.addPrimitive(base.sub, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, 'sub')
    pset.addPrimitive(base.sub, [OperatorType, OperatorType], OperatorType, 'sub')

    pset.addPrimitive(base.mul, [DiagonalOperatorType, DiagonalOperatorType], DiagonalOperatorType, 'mul')
    pset.addPrimitive(base.mul, [BlockDiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, 'mul')
    pset.addPrimitive(base.mul, [DiagonalOperatorType, BlockDiagonalOperatorType], BlockDiagonalOperatorType, 'mul')
    pset.addPrimitive(base.mul, [BlockDiagonalOperatorType, DiagonalOperatorType], BlockDiagonalOperatorType, 'mul')
    pset.addPrimitive(base.mul, [OperatorType, OperatorType], OperatorType, 'mul')

    pset.addPrimitive(base.minus, [OperatorType], OperatorType, 'minus')

    pset.addPrimitive(base.inv, [DiagonalOperatorType], DiagonalOperatorType, 'inverse')
    pset.addPrimitive(base.inv, [BlockDiagonalOperatorType], OperatorType, 'inverse')

    pset.addPrimitive(base.mul, [OperatorType, GridType], GridType, 'mul')


    # Correction
    import functools
    residual = functools.partial(mg.residual, terminals.grid, terminals.operator)
    pset.addPrimitive(residual, [GridType], GridType, 'residual')
    correct = functools.partial(mg.correct, terminals.grid)
    pset.addPrimitive(correct, [GridType, part.Partitioning], GridType, 'correct')

    # Multigrid recipes
    CoarseGridType = types.CoarseGrid
    CoarseOperatorType = types.CoarseOperator
    InterpolationType = types.Interpolation
    RestrictionType = types.Restriction

    # Create intergrid operators
    pset.addPrimitive(base.mul, [CoarseOperatorType, RestrictionType], RestrictionType, 'mul')
    pset.addPrimitive(base.mul, [InterpolationType, CoarseOperatorType], InterpolationType, 'mul')
    pset.addPrimitive(base.mul, [InterpolationType, RestrictionType], OperatorType, 'mul')

    pset.addPrimitive(base.mul, [RestrictionType, GridType], CoarseGridType, 'mul')
    pset.addPrimitive(base.mul, [InterpolationType, CoarseGridType], GridType, 'mul')
    pset.addPrimitive(base.mul, [CoarseOperatorType, CoarseGridType], CoarseGridType, 'mul')

    pset.addPrimitive(lambda x: x, [CoarseOperatorType], CoarseOperatorType, 'noop')
    pset.addPrimitive(lambda x: x, [RestrictionType], RestrictionType, 'noop')
    pset.addPrimitive(lambda x: x, [InterpolationType], InterpolationType, 'noop')
    pset.addPrimitive(lambda x: x, [part.Partitioning], part.Partitioning, 'noop')


def generate_multigrid(operator, grid, rhs, dimension, coarsening_factor,
                       interpolation_stencil=None, restriction_stencil=None, maximum_number_of_cycles=1):
    assert maximum_number_of_cycles >= 1, "The maximum number of cycles must be greater zero"
    terminals = Terminals(operator, grid, rhs, dimension, coarsening_factor, interpolation_stencil, restriction_stencil)
    types = Types(terminals)
    pset = gp.PrimitiveSetTyped("main", [], types.Grid)
    pset.addTerminal(rhs, types.Grid, 'f')
    add_multigrid_cycle(pset, terminals, types)
    for _ in range(1, maximum_number_of_cycles):
        coarse_grid = base.ZeroGrid(terminals.coarse_grid.size)
        terminals = Terminals(terminals.coarse_operator, coarse_grid, dimension, coarsening_factor,
                              interpolation_stencil, restriction_stencil)
        add_multigrid_cycle(pset, terminals)

    return pset