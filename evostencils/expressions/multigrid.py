import sympy as sp


def smooth(grid, smoother, operator, rhs):
    A = operator
    u = grid
    f = rhs
    B = smoother
    return u + B * (f - A * u)


def residual(grid, operator, rhs):
    return rhs - operator * grid


def correct(grid, interpolation, correction):
    return grid + interpolation * correction


def get_interpolation(fine_grid, coarse_grid):
    return sp.MatrixSymbol(f'I_{coarse_grid.shape[0]}', coarse_grid.shape[0], fine_grid.shape[1])


def get_restriction(fine_grid, coarse_grid):
    return sp.MatrixSymbol(f'R_{fine_grid.shape[0]}', fine_grid.shape[0], coarse_grid.shape[1])


def get_coarse_grid(fine_grid, factor):
    return sp.MatrixSymbol(f'{fine_grid.name}_{fine_grid.shape[0]}', fine_grid.shape[0] / factor, fine_grid.shape[1])


def get_coarse_operator(fine_operator, factor):
    return sp.MatrixSymbol(f'{fine_operator.name}_{fine_operator.shape[0]}', fine_operator.shape[0] / factor, fine_operator.shape[1] / factor)
