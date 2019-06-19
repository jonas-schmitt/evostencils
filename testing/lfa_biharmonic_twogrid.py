# Smoothing analysis of the biharmonic equation

from lfa_lab import *

grid = Grid(2)
coarse_grid = grid.coarse((2,2,))
Laplace = gallery.poisson_2d(grid)
I = operator.identity(grid)
Z = operator.zero(grid)

A = system([[Laplace, I],
            [Z      , Laplace]])
S_pointwise = jacobi(A, 0.8)
S_collective = collective_jacobi(A, 0.8)


Rs = gallery.fw_restriction(grid, coarse_grid)
Ps = gallery.ml_interpolation(grid, coarse_grid)

RZ = Rs.matching_zero()
PZ = Ps.matching_zero()

R = system([[Rs, RZ],
            [RZ, Rs]])
P = system([[Ps, PZ],
            [PZ, Ps]])

# Create the Galerkin coarse grid approximation
Ac = R * A * P

cgc = coarse_grid_correction(
        operator = A,
        coarse_operator = Ac,
        interpolation = P,
        restriction = R)

#E = S_pointwise * cgc * S_pointwise
# E = S_collective * cgc * S_collective
E = S_pointwise
symbol = E.symbol()

print("Spectral radius: {}".format(symbol.spectral_radius()))
# print("Spectral norm: {}".format(symbol.spectral_norm()))

