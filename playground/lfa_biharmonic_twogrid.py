# Smoothing analysis of the biharmonic equation
import evostencils.evaluation.convergence as convergence
import evostencils.stencils.constant as constant
import evostencils.expressions.base as base

from lfa_lab import *
import lfa_lab
import lfa_lab.block_smoother

grid = Grid(2, [1.0/32, 1.0/32])
coarse_grid = grid.coarse((2,2,))
a = [ ((-1,0),-1), ((0, -1), -1),
      ((0,0), 4), ((0,1), -1), ((1,0), -1) ]
Laplace = operator.from_stencil(a, grid)

I = operator.identity(grid)
Z = operator.zero(grid)

A = system([[Laplace, I],
            [Z      , Laplace]])
S_pointwise = jacobi(A, 0.6)
S_collective = collective_jacobi(A, 0.6)

block_size = np.array((2, 2))

d = NdArray(shape=(1,2))
d[0,0] = [ ((0,-1), -1), ((0,0), 4), ((0,1), -1)]
d[0,1] = [ ((0,-1), -1), ((0,0), 4), ((0,1), -1)]
#d[1,0] = [ ((-1,0), -1), ((0,0), 4), ((0,1), -1) ]
#d[1,1] = [ ((-1,0), -1), ((0,-1), -1), ((0,0), 4) ]
#d[0,0] = [ ((0,0), 4)]
#d[0,1] = [ ((0,0), 4)]
#d[1,0] = [ ((0,0), 4) ]
#d[1,1] = [ ((0,0), 4) ]
D = operator.from_periodic_stencil(d, grid)

d = NdArray(shape=(1,2))
d[0,0] = [ ((0,0), 1)]
d[0,1] = [ ((0,0), 1)]
#d[1,0] = [ ((0,0), 1)]
#d[1,1] = [ ((0,0), 1)]
I_bl = operator.from_periodic_stencil(d, grid)

d = NdArray(shape=(1,2))
Z_bl = operator.from_periodic_stencil(d, grid)



tmp = system([[D, I_bl], [Z_bl, D]])
tmp = tmp.matching_identity() - 0.6 * tmp.inverse() * A



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
# E = tmp * cgc * tmp
# E = S_collective * cgc * S_collective
# E = S_pointwise
E = tmp
symbol = E.symbol()

print("Spectral radius: {}".format(symbol.spectral_radius()))
# print("Spectral norm: {}".format(symbol.spectral_norm()))

