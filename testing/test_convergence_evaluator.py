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
A = base.generate_operator('A', fine_grid_size, constant.Stencil(entries))

# Create a poisson operator.
fine_operator = lfa.gallery.poisson_2d(grid)
coarsening_factor = (2, 2)
u_coarse = multigrid.get_coarse_grid(u, coarsening_factor)
A_coarse = multigrid.get_coarse_operator(A, u_coarse)
P = multigrid.get_interpolation(u, u_coarse)
R = multigrid.get_restriction(u, u_coarse)

coarse_operator = lfa.gallery.poisson_2d(grid.coarse(coarsening_factor))
evaluator = ConvergenceEvaluator(grid, coarsening_factor, len(coarsening_factor), lfa.gallery.ml_interpolation, lfa.gallery.fw_restriction)
smoother = base.Inverse(base.Diagonal(A))
correction = base.mul(smoother, multigrid.residual(u, A, b))
tmp = multigrid.cycle(u, correction, partitioning=partitioning.Single, weight=1)
iteration_matrix = Optimizer.get_iteration_matrix(tmp, u, b)
print(iteration_matrix)
print(evaluator.transform(iteration_matrix, evaluator.grid).symbol().spectral_radius())
