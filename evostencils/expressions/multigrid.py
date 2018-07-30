from evostencils.expressions import base


class Restriction(base.Operator):
    def __init__(self, grid, coarse_grid):
        super(Restriction, self).__init__(f'R_{grid.size}', (coarse_grid.size, grid.size))


class Interpolation(base.Operator):
    def __init__(self, grid, coarse_grid):
        super(Interpolation, self).__init__(f'I_{coarse_grid.size}', (grid.size, coarse_grid.size))


def correct(iteration_matrix, grid, operator, rhs):
    A = operator
    u = grid
    f = rhs
    B = iteration_matrix
    return base.Addition(u, base.Multiplication(B, residual(u, A, f)))


def residual(grid, operator, rhs):
    return base.Subtraction(rhs, base.Multiplication(operator, grid))


def get_interpolation(grid, coarse_grid):
    return Interpolation(grid, coarse_grid)


def get_restriction(grid, coarse_grid):
    return Restriction(grid, coarse_grid)


def get_coarse_grid(grid, coarsening_factor):
    return base.Grid(f'{grid.name}_coarse', grid.size / coarsening_factor)


def get_coarse_operator(operator, coarsening_factor):
    return base.Operator(f'{operator.name}_coarse', (operator.shape[0] / coarsening_factor,
                                                     operator.shape[1] / coarsening_factor))

