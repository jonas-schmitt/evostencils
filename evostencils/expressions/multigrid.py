import sympy as sp

def smooth(grid, smoother, operator, rhs):
    A = operator
    u = grid
    f = rhs
    B = smoother
    return u + B * (f - A * u)

def residual(grid, operator, rhs):
    return rhs - operator * grid
