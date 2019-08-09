# Smoothing analysis of the biharmonic equation
import evostencils.evaluation.convergence as convergence
import evostencils.stencils.constant as constant
import evostencils.expressions.base as base

from lfa_lab import *
import lfa_lab
import lfa_lab.block_smoother

grid = Grid(2)
coarse_grid = grid.coarse((2,2,))
Laplace = gallery.poisson_2d(grid)
I = operator.identity(grid)
Z = operator.zero(grid)

A = system([[Laplace, I],
            [Z      , Laplace]])
S_pointwise = jacobi(A, 0.6)
S_collective = collective_jacobi(A, 0.6)

block_size = np.array((2, 2))

diag_stencil = lfa_lab.block_smoother._block_diag_stencil(Laplace.stencil, block_size)
D = from_periodic_stencil(diag_stencil, grid)
I = A.matching_identity()
entries = [
  ]
Z_periodic = SparseStencil(entries)
diag_stencil = lfa_lab.block_smoother._block_diag_stencil(Z_periodic, block_size)
Z_periodic = from_periodic_stencil(diag_stencil, grid)
entries = [
    (( 0,  0),  1)
  ]
I_periodic = SparseStencil(entries)
diag_stencil = lfa_lab.block_smoother._block_diag_stencil(I_periodic, block_size)
I_periodic = from_periodic_stencil(diag_stencil, grid)


tmp = system([[D,I_periodic], [Z_periodic, D]])
tmp = I - 0.6 * tmp.inverse() * A



Rs = gallery.fw_restriction(grid, coarse_grid)
Ps = gallery.ml_interpolation(grid, coarse_grid)

RZ = Rs.matching_zero()
PZ = Ps.matching_zero()

R = system([[Rs, RZ],
            [RZ, Rs]])
P = system([[Ps, PZ],
            [PZ, Ps]])

# Create the Galerkin coarse grid approximation
#Ac = R * A * P
Lc = gallery.poisson_2d(coarse_grid)
Ac = system([[Lc, Lc.matching_identity()], [Lc.matching_zero(), Lc]])

cgc = coarse_grid_correction(
        operator = A,
        coarse_operator = Ac,
        interpolation = P,
        restriction = R)

#E = cgc
# E = S_pointwise * cgc * S_pointwise
E = tmp * cgc * tmp
# E = S_collective * cgc * S_collective
# E = S_pointwise
symbol = E.symbol()

print("Spectral radius: {}".format(symbol.spectral_radius()))
# print("Spectral norm: {}".format(symbol.spectral_norm()))

