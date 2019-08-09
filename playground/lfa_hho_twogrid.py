# Smoothing analysis of a hho stencil

from lfa_lab import *

h1, h2 = 1.0/32, 1.0/32
grid = Grid(2)
coarse_grid = grid.coarse((2, 2))
entries = [((0, 0), 5.0/2.0),
           ((-1, 0), -3.0/4.0),
           ((1, 0), -3/4.0)
           ]
A_00 = operator.from_stencil(entries, grid)

entries = [((0, 0), 5.0/2.0),
           ((0, -1), -3.0/4.0),
           ((0, 1), -3/4.0)
           ]
A_11 = operator.from_stencil(entries, grid)

entries = [((0, 0), -1.0/4.0),
           ((-1, 0), -1.0/4.0),
           ((-1, 1), -1.0/4.0),
           ((0, 1), -1.0/4.0)
           ]
A_10 = operator.from_stencil(entries, grid)

entries = [((0, 0), -1.0/4.0),
           ((0, -1), -1.0/4.0),
           ((1, -1), -1.0/4.0),
           ((1, 0), -1.0/4.0)
           ]

A_01 = operator.from_stencil(entries, grid)

A = system([[A_00, A_10], [A_01, A_11]])
S_pointwise = jacobi(A, 1)
S_collective = collective_jacobi(A, 1)


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

# E = S_pointwise * cgc * S_pointwise
E = S_collective * cgc * S_collective
symbol = E.symbol()

print("Spectral radius: {}".format(symbol.spectral_radius()))
# print("Spectral norm: {}".format(symbol.spectral_norm()))

