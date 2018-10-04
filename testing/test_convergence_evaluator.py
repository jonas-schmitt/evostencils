from evostencils.expressions import base, multigrid, partitioning
from evostencils.expressions import transformations
from evostencils.optimizer import Optimizer
import lfa_lab as lfa
from evostencils.stencils import constant, gallery



from evostencils.evaluation.convergence import *
# Create a 2D grid with step-size (1/32, 1/32).
dimension = 2
grid_size = (512, 512)
step_size = (1.0, 1.0)
coarsening_factor = (2, 2)

lfa_grid = lfa.Grid(dimension, step_size)
lfa_operator = lfa.gallery.poisson_2d(lfa_grid)
lfa_coarse_operator = lfa.gallery.poisson_2d(lfa_grid.coarse(coarsening_factor))

u = base.generate_grid('u', grid_size, step_size)
b = base.generate_grid('f', grid_size, step_size)
A = base.generate_operator_on_grid('A', u, gallery.generate_poisson_2d)

u_coarse = multigrid.get_coarse_grid(u, coarsening_factor)
A_coarse = multigrid.get_coarse_operator(A, u_coarse)
P = multigrid.get_interpolation(u, u_coarse)
R = multigrid.get_restriction(u, u_coarse)

evaluator = ConvergenceEvaluator(lfa_grid, coarsening_factor, dimension, lfa.gallery.poisson_2d, lfa.gallery.ml_interpolation, lfa.gallery.fw_restriction)
# Jacobi
smoother = base.Inverse(base.Diagonal(A))
correction = base.mul(smoother, multigrid.residual(u, A, b))
jacobi = multigrid.cycle(u, correction, partitioning=partitioning.Single, weight=1)
iteration_matrix = Optimizer.get_iteration_matrix(jacobi, u, b)
#print(iteration_matrix)
#print(evaluator.transform(iteration_matrix, evaluator.grid).symbol().spectral_radius())

# Block-Jacobi
smoother = base.Inverse(base.BlockDiagonal(A, (2, 2)))
correction = base.mul(smoother, multigrid.residual(u, A, b))
block_jacobi = multigrid.cycle(u, correction, partitioning=partitioning.Single, weight=1)
iteration_matrix = Optimizer.get_iteration_matrix(block_jacobi, u, b)
#print(iteration_matrix)
#print(evaluator.transform(iteration_matrix, evaluator.grid).symbol().spectral_radius())

# Red-Black-Block-Jacobi
smoother = base.Inverse(base.Diagonal(A))
correction = base.mul(smoother, multigrid.residual(u, A, b))
rb_jacobi = multigrid.cycle(u, correction, partitioning=partitioning.RedBlack, weight=1)
iteration_matrix = Optimizer.get_iteration_matrix(rb_jacobi, u, b)
#print(iteration_matrix)
#print(evaluator.transform(iteration_matrix, evaluator.grid).symbol().spectral_radius())

# Two-Grid
tmp = multigrid.residual(u, A, b)
tmp = base.mul(multigrid.Restriction(u, u_coarse), tmp)
zero = base.ZeroGrid(u_coarse.size, u_coarse.step_size)
f = tmp
tmp = multigrid.residual(zero, A_coarse, f)
tmp = multigrid.cycle(zero, base.mul(base.Inverse(base.Diagonal(A_coarse)), tmp))
tmp = multigrid.residual(tmp, A_coarse, f)
u_coarse_coarse = multigrid.get_coarse_grid(u_coarse, coarsening_factor)
A_coarse_coarse = multigrid.get_coarse_operator(A_coarse, u_coarse_coarse)
tmp = base.mul(multigrid.Restriction(u_coarse, u_coarse_coarse), tmp)
tmp = base.mul(multigrid.CoarseGridSolver(A_coarse_coarse), tmp)
tmp = base.mul(multigrid.Interpolation(u_coarse, u_coarse_coarse), tmp)
tmp = multigrid.cycle(zero, tmp)
tmp = base.mul(multigrid.Interpolation(u, u_coarse), tmp)
tmp = multigrid.cycle(jacobi, tmp)

iteration_matrix = Optimizer.get_iteration_matrix(tmp, u, b)
print(iteration_matrix)
print(evaluator.compute_spectral_radius(iteration_matrix))

L = lfa.gallery.poisson_2d(lfa_grid)
Lc = lfa.gallery.poisson_2d(lfa_grid.coarse(coarsening_factor))
S = lfa.jacobi(L, 1)
RB = lfa.rb_jacobi(L, 1)

# Create restriction and interpolation operators.
restriction = lfa.gallery.fw_restriction(lfa_grid, lfa_grid.coarse(coarsening_factor))
interpolation = lfa.gallery.ml_interpolation(lfa_grid, lfa_grid.coarse(coarsening_factor))

# Construct the coarse grid correction from the individual operators.
cgc = lfa.coarse_grid_correction(
        operator = L,
        coarse_operator = Lc,
        interpolation = interpolation,
        restriction = restriction)

# Apply one pre- and one post-smoothing step.
E = cgc

print(E.symbol().spectral_radius())

I = lfa.identity(lfa_grid)
R = lfa.gallery.fw_restriction(lfa_grid, lfa_grid.coarse(coarsening_factor))
R_c = lfa.gallery.fw_restriction(lfa_grid.coarse(coarsening_factor), lfa_grid.coarse(coarsening_factor).coarse(coarsening_factor))
P = lfa.gallery.ml_interpolation(lfa_grid, lfa_grid.coarse(coarsening_factor))
P_c = lfa.gallery.ml_interpolation(lfa_grid.coarse(coarsening_factor), lfa_grid.coarse(coarsening_factor).coarse(coarsening_factor))
A = lfa.gallery.poisson_2d(lfa_grid)
A_cc = lfa.gallery.poisson_2d(lfa_grid.coarse(coarsening_factor).coarse(coarsening_factor))
zero = lfa.zero(lfa_grid)
zero_c = lfa.zero(lfa_grid.coarse(coarsening_factor))
zero_cc = lfa.zero(lfa_grid.coarse(coarsening_factor).coarse(coarsening_factor))
mg = (lfa.jacobi(A) + 1.0 * (P * (1.0 * (P_c * (A_cc.inverse() * (R_c * (R * (-1.0 * (A * I)))))))))
print(mg.symbol().spectral_radius())

