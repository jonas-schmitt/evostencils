from evostencils.expressions import base, multigrid, partitioning
from evostencils.expressions import transformations
from evostencils.optimizer import Optimizer
import lfa_lab as lfa
import evostencils.stencils.constant as constant
import evostencils.stencils.periodic as periodic


from evostencils.evaluation.convergence import *
# Create a 2D grid with step-size (1/32, 1/32).
grid = lfa.Grid(2, [1.0, 1.0])
entries = [
        (( 0, -1), -1.0),
        ((-1,  0), -1.0),
        (( 0,  0),  4.0),
        (( 1,  0), -1.0),
        (( 0,  1), -1.0)
    ]

fine_grid_size = (64, 64)

u = base.generate_grid('x', fine_grid_size)
b = base.generate_grid('b', fine_grid_size)
A = base.generate_operator_on_grid('A', fine_grid_size, constant.Stencil(entries))

# Create a poisson operator.
fine_operator = lfa.gallery.poisson_2d(grid)
coarsening_factor = (2, 2)
u_coarse = multigrid.get_coarse_grid(u, coarsening_factor)
A_coarse = multigrid.get_coarse_operator(A, u_coarse)
P = multigrid.get_interpolation(u, u_coarse)
R = multigrid.get_restriction(u, u_coarse)

coarse_operator = lfa.gallery.poisson_2d(grid.coarse(coarsening_factor))
evaluator = ConvergenceEvaluator(grid, coarsening_factor, len(coarsening_factor), lfa.gallery.poisson_2d, lfa.gallery.ml_interpolation, lfa.gallery.fw_restriction)
# Jacobi
smoother = base.Inverse(base.Diagonal(A))
correction = base.mul(smoother, multigrid.residual(u, A, b))
jacobi = multigrid.cycle(u, correction, partitioning=partitioning.Single, weight=0.8)
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
smoother = base.Inverse(base.BlockDiagonal(A, (2, 2)))
correction = base.mul(smoother, multigrid.residual(u, A, b))
rb_jacobi = multigrid.cycle(u, correction, partitioning=partitioning.RedBlack, weight=1)
iteration_matrix = Optimizer.get_iteration_matrix(rb_jacobi, u, b)
#print(iteration_matrix)
#print(evaluator.transform(iteration_matrix, evaluator.grid).symbol().spectral_radius())

# Two-Grid
tmp = multigrid.residual(u, A, b)
tmp = base.mul(multigrid.Restriction(u, u_coarse), tmp)
tmp = base.mul(multigrid.CoarseGridSolver(A_coarse), tmp)
tmp = base.mul(multigrid.Interpolation(u, u_coarse), tmp)
tmp = multigrid.cycle(u, tmp)
iteration_matrix = Optimizer.get_iteration_matrix(tmp, u, b)
print(iteration_matrix)
print(evaluator.compute_spectral_radius(iteration_matrix))

L = lfa.gallery.poisson_2d(grid)
Lc = lfa.gallery.poisson_2d(grid.coarse(coarsening_factor))
S = lfa.jacobi(L, 0.8)

# Create restriction and interpolation operators.
restriction = lfa.gallery.fw_restriction(grid, grid.coarse(coarsening_factor))
interpolation = lfa.gallery.ml_interpolation(grid, grid.coarse(coarsening_factor))

# Construct the coarse grid correction from the individual operators.
cgc = lfa.coarse_grid_correction(
        operator = L,
        coarse_operator = Lc,
        interpolation = interpolation,
        restriction = restriction)

# Apply one pre- and one post-smoothing step.
E = cgc

print(E.symbol().spectral_radius())
