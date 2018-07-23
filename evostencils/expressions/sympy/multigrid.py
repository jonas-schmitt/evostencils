import sympy as sp
from evostencils.expressions import block


class Restriction(sp.MatrixExpr):
    def __new__(cls, m, n):
        return super(Restriction, cls).__new__(cls, m, n)

    @property
    def shape(self):
        return self.args[0], self.args[1]


class Interpolation(sp.MatrixExpr):
    def __new__(cls, m, n):
        return super(Interpolation, cls).__new__(cls, m, n)

    @property
    def shape(self):
        return self.args[0], self.args[1]


def correct(iteration_matrix, grid, operator, rhs):
    A = operator
    u = grid
    f = rhs
    B = iteration_matrix
    return u + B * residual(u, A, f)


def residual(grid, operator, rhs):
    return rhs - operator * grid


def get_interpolation(fine_grid, coarse_grid):
    return Interpolation(fine_grid.shape[0], coarse_grid.shape[0])


def get_restriction(fine_grid, coarse_grid):
    return Restriction(coarse_grid.shape[0], fine_grid.shape[0])


def get_coarse_grid(fine_grid, factor):
    return sp.MatrixSymbol(f'{str(fine_grid)}_coarse', fine_grid.shape[0] / factor, fine_grid.shape[1])


def get_coarse_operator(fine_operator, factor):
    return sp.MatrixSymbol(f'{str(fine_operator)}_coarse', fine_operator.shape[0] / factor, fine_operator.shape[1] / factor)


def coarse_grid_correction(grid, operator, rhs, coarse_operator, restriction, interpolation, coarse_error=None):
    u = grid
    f = rhs
    I = sp.Identity(operator.shape[1])
    Ic = sp.Identity(coarse_operator.shape[1])
    P = interpolation
    R = restriction
    L = operator
    Lc = coarse_operator
    Ec = coarse_error
    if coarse_error is None:
        Ec = sp.ZeroMatrix(*Ic.shape)
    return u + P * (Ic - Ec) * Lc.I * R * (f - L * u)

