from evostencils.expressions import base, multigrid, partitioning
from evostencils.stencils import gallery
from evostencils.exastencils.declarations import *

dimension = 2
grid_size = (512, 512)
step_size = (1.0, 1.0)
coarsening_factor = (2, 2)


u = base.generate_grid('u', grid_size, step_size)
b = base.generate_rhs('f', grid_size, step_size)
A = base.generate_operator_on_grid('A', u, gallery.generate_poisson_2d)

u_coarse = multigrid.get_coarse_grid(u, coarsening_factor)
A_coarse = multigrid.get_coarse_operator(A, u_coarse)
P = multigrid.get_interpolation(u, u_coarse)
R = multigrid.get_restriction(u, u_coarse)

# Jacobi
smoother = base.Inverse(base.Diagonal(A))
correction = base.mul(smoother, multigrid.residual(A, u, b))
jacobi = multigrid.cycle(u, None, correction, partitioning=partitioning.Single, weight=1)
declarations = identify_temporary_fields(jacobi)

# Block-Jacobi
smoother = base.Inverse(base.BlockDiagonal(A, (2, 2)))
correction = base.mul(smoother, multigrid.residual(A, u, b))
block_jacobi = multigrid.cycle(u, None, correction, partitioning=partitioning.Single, weight=1)
declarations = identify_temporary_fields(block_jacobi)

# Red-Black-Block-Jacobi
smoother = base.Inverse(base.Diagonal(A))
correction = base.mul(smoother, multigrid.residual(A, u, b))
rb_jacobi = multigrid.cycle(u, None, correction, partitioning=partitioning.RedBlack, weight=1)
declarations = identify_temporary_fields(rb_jacobi)

# Two-Grid
tmp = multigrid.residual(A, u, b)
tmp = base.mul(multigrid.Restriction(u, u_coarse), tmp)
tmp = base.mul(multigrid.CoarseGridSolver(A_coarse), tmp)
tmp = base.mul(multigrid.Interpolation(u, u_coarse), tmp)
tmp = multigrid.cycle(u, None, tmp)
declarations = identify_temporary_fields(tmp)

# Three-Grid
tmp = multigrid.residual(A, u, b)
tmp = base.mul(multigrid.Restriction(u, u_coarse), tmp)
zero = base.ZeroGrid(u_coarse.size, u_coarse.step_size)
f = tmp
tmp = multigrid.residual(A_coarse, zero, f)
#tmp = multigrid.cycle(zero, None, base.mul(base.Inverse(base.Diagonal(A_coarse)), tmp))
#tmp = multigrid.residual(A_coarse, tmp, f)
u_coarse_coarse = multigrid.get_coarse_grid(u_coarse, coarsening_factor)
A_coarse_coarse = multigrid.get_coarse_operator(A_coarse, u_coarse_coarse)
tmp = base.mul(multigrid.Restriction(u_coarse, u_coarse_coarse), tmp)
tmp = base.mul(multigrid.CoarseGridSolver(A_coarse_coarse), tmp)
tmp = base.mul(multigrid.Interpolation(u_coarse, u_coarse_coarse), tmp)
tmp = multigrid.cycle(zero, None, tmp)
tmp = base.mul(multigrid.Interpolation(u, u_coarse), tmp)
tmp = multigrid.cycle(u, None, tmp)
#tmp = multigrid.cycle(jacobi, None, tmp)
declarations = identify_temporary_fields(tmp)
name_fields(declarations)
print_declarations(declarations)
print(tmp)
