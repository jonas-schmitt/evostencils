from evostencils.expressions import base, multigrid, partitioning
from evostencils.stencils import gallery
from evostencils.exastencils.generation import *

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
generator = ProgramGenerator(A, u, b, dimension, coarsening_factor, P, R)
# Jacobi
smoother = base.Inverse(base.Diagonal(A))
correction = base.mul(smoother, multigrid.residual(A, u, b))
jacobi = multigrid.cycle(u, b, correction, partitioning=partitioning.Single, weight=1)
print("Generating Jacobi\n")
print(generator.generate(jacobi))

# Block-Jacobi
smoother = base.Inverse(base.BlockDiagonal(A, (2, 2)))
correction = base.mul(smoother, multigrid.residual(A, u, b))
block_jacobi = multigrid.cycle(u, b, correction, partitioning=partitioning.Single, weight=1)

# Red-Black-Block-Jacobi
smoother = base.Inverse(base.Diagonal(A))
correction = base.mul(smoother, multigrid.residual(A, u, b))
rb_jacobi = multigrid.cycle(u, b, correction, partitioning=partitioning.RedBlack, weight=1)

# Two-Grid
#tmp = multigrid.residual(A, u, b)
#tmp = base.mul(multigrid.get_restriction(u, u_coarse), tmp)
#tmp = base.mul(multigrid.CoarseGridSolver(A_coarse), tmp)
#tmp = base.mul(multigrid.get_interpolation(u, u_coarse), tmp)
#tmp = multigrid.cycle(u, None, tmp)

zero = base.ZeroGrid(u_coarse.size, u_coarse.step_size)
# Three-Grid
tmp = mg.cycle(u, b, base.mul(mg.get_restriction(u, u_coarse), mg.residual(A, u, b)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(zero, tmp.correction, mg.residual(A_coarse, zero, tmp.correction)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(tmp.correction.iterate, tmp.correction.rhs, base.mul(base.Inverse(base.Diagonal(A_coarse)), tmp.correction.correction)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(tmp.correction, tmp.correction.rhs, mg.residual(A_coarse, tmp.correction, tmp.correction.rhs)))
u_coarse_coarse = multigrid.get_coarse_grid(u_coarse, coarsening_factor)
A_coarse_coarse = multigrid.get_coarse_operator(A_coarse, u_coarse_coarse)
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(tmp.correction.iterate, tmp.correction.rhs, base.mul(mg.get_restriction(u_coarse, u_coarse_coarse), tmp.correction.correction)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(tmp.correction.iterate, tmp.correction.rhs, base.mul(mg.get_coarse_grid_solver(A_coarse_coarse), tmp.correction.correction)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(tmp.correction.iterate, tmp.correction.rhs, base.mul(mg.get_interpolation(u_coarse, u_coarse_coarse), tmp.correction.correction)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, base.mul(mg.get_interpolation(u, u_coarse), tmp.correction))
"""
tmp = multigrid.residual(A, u, b)
tmp = base.mul(multigrid.get_restriction(u, u_coarse), tmp)
f = tmp
tmp = multigrid.cycle(zero, f, base.mul(base.Inverse(base.Diagonal(A_coarse)), mg.residual(A_coarse, zero, tmp)))
rhs = tmp.rhs
tmp = multigrid.residual(A_coarse, tmp, tmp.rhs)
u_coarse_coarse = multigrid.get_coarse_grid(u_coarse, coarsening_factor)
A_coarse_coarse = multigrid.get_coarse_operator(A_coarse, u_coarse_coarse)
tmp = base.mul(multigrid.get_restriction(u_coarse, u_coarse_coarse), tmp)
tmp = base.mul(multigrid.CoarseGridSolver(A_coarse_coarse), tmp)
tmp = base.mul(multigrid.get_interpolation(u_coarse, u_coarse_coarse), tmp)
tmp = multigrid.cycle(zero, rhs, tmp)
tmp = base.mul(multigrid.get_interpolation(u, u_coarse), tmp)
tmp = multigrid.cycle(u, b, tmp)
#tmp = multigrid.cycle(jacobi, None, tmp)
"""
print("Generating Multigrid\n")
print(generator.generate(tmp))
#print_declarations(temporaries)
