import sympy as sp


class Interpolation(sp.MatrixSymbol):
    pass


class Restriction(sp.MatrixSymbol):
    pass


def smooth(grid, smoother, operator, rhs):
    A = operator
    u = grid
    f = rhs
    B = smoother
    return u + B * (f - A * u)


def residual(grid, operator, rhs):
    return rhs - operator * grid


def correct(grid, correction):
    return grid + correction


def get_interpolation(coarse_grid, fine_grid):
    return Interpolation(f'I_{coarse_grid.shape[0]}', fine_grid.shape[0], coarse_grid.shape[0])


def get_restriction(fine_grid, coarse_grid):
    return Restriction(f'R_{fine_grid.shape[0]}', coarse_grid.shape[0], fine_grid.shape[0])


def get_coarse_grid(fine_grid, factor):
    return sp.MatrixSymbol(f'{str(fine_grid)}_{fine_grid.shape[0]}', fine_grid.shape[0] / factor, fine_grid.shape[1])


def get_coarse_operator(fine_operator, factor):
    return sp.MatrixSymbol(f'{str(fine_operator)}_{fine_operator.shape[0] / factor}', fine_operator.shape[0] / factor, fine_operator.shape[1] / factor)



