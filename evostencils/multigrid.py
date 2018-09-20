from evostencils.expressions import multigrid as mg
from evostencils.expressions import base
from evostencils.expressions import partitioning as part
from evostencils.stencils import constant
from evostencils import matrix_types


class Terminals:
    def __init__(self, operator, grid, rhs, dimension, coarsening_factor, interpolation_stencil=None, restriction_stencil=None):
        self.operator = operator
        self.grid = grid
        self.rhs = rhs
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
        self.zero_matrix = base.Zero(self.operator.shape)
        self.coarse_grid_solver = mg.CoarseGridSolver(self.coarse_grid)
        self.no_partitioning = part.Single
        self.red_black_partitioning = part.RedBlack


class Types:
    def __init__(self, terminals):
        self.Operator = matrix_types.generate_matrix_type(terminals.operator.shape)
        self.Grid = matrix_types.generate_matrix_type(terminals.grid.shape)
        self.DiagonalOperator = matrix_types.generate_diagonal_matrix_type(terminals.diagonal.shape)
        self.BlockDiagonalOperator = matrix_types.generate_block_diagonal_matrix_type(terminals.block_diagonal)
        self.Interpolation = matrix_types.generate_matrix_type(terminals.interpolation.shape)
        self.Restriction = matrix_types.generate_matrix_type(terminals.restriction.shape)
        self.CoarseOperator = matrix_types.generate_matrix_type(terminals.coarse_operator.shape)
        self.CoarseGrid = matrix_types.generate_matrix_type(terminals.coarse_grid.shape)

