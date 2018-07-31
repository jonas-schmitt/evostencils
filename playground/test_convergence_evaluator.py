from evostencils.expressions import base, multigrid
from evostencils.expressions import transformations
from evostencils.optimizer import Optimizer
import sympy as sp
import lfa_lab as lfa

fine_grid_size = (64, 64)

u = base.generate_grid('x', fine_grid_size)
b = base.generate_grid('b', fine_grid_size)
A = base.generate_operator('A', fine_grid_size)

from evostencils.evaluation.convergence import *
# Create a 2D grid with step-size (1/32, 1/32).
fine = lfa.Grid(2, [1.0, 1.0])

# Create a poisson operator.
fine_operator = lfa.gallery.poisson_2d(fine)

coarsening_factor = 4
u_coarse = multigrid.get_coarse_grid(u, coarsening_factor)
A_coarse = multigrid.get_coarse_operator(A, coarsening_factor)
P = multigrid.get_interpolation(u, u_coarse)
R = multigrid.get_restriction(u, u_coarse)

coarse_operator = lfa.gallery.poisson_2d(fine.coarse((2, 2)))
evaluator = ConvergenceEvaluator(fine_operator, coarse_operator, fine, fine_grid_size, (2, 2))
smoother = base.Inverse(base.Diagonal(A))
tmp = multigrid.correct(smoother, u, A, b)
coarse_grid_correction = base.Multiplication(P, base.Multiplication(base.Inverse(A_coarse), R))
tmp = multigrid.correct(coarse_grid_correction, tmp, A, b)
two_grid = multigrid.correct(smoother, tmp, A, b)
iteration_matrix = Optimizer.get_iteration_matrix(two_grid, u, b)
print(iteration_matrix)
print(evaluator.transform(iteration_matrix).symbol().spectral_radius())

jacobi = lfa_lab.jacobi(fine_operator, 1.0)
reference = jacobi * lfa_lab.coarse_grid_correction(fine_operator, coarse_operator, evaluator.interpolation, evaluator.restriction, coarse_error=None) * jacobi
print(reference.symbol().spectral_radius())


