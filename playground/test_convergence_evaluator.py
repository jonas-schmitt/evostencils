from evostencils.expressions import block, multigrid
import lfa_lab as lfa

fine_grid_size = (64, 64)

u = block.generate_vector_on_grid('x', fine_grid_size)
b = block.generate_vector_on_grid('b', fine_grid_size)
A = block.generate_matrix_on_grid('A', fine_grid_size)

from evostencils.evaluation.convergence import *
# Create a 2D grid with step-size (1/32, 1/32).
fine = lfa.Grid(2, [1.0, 1.0])

# Create a poisson operator.
fine_operator = lfa.gallery.poisson_2d(fine)

coarsening_factor = 4
u_coarse = multigrid.get_coarse_grid(u, coarsening_factor)
A_coarse = multigrid.get_coarse_operator(A, coarsening_factor)
P = multigrid.get_interpolation(u_coarse, u)
R = multigrid.get_restriction(u, u_coarse)

coarse_operator = lfa.gallery.poisson_2d(fine.coarse((2, 2)))
evaluator = ConvergenceEvaluator(fine_operator, coarse_operator, fine, fine_grid_size, (2, 2))
smoother = sp.Identity(A.shape[0]) - block.get_diagonal(A).I * A * sp.Identity(A.shape[0])
two_grid = smoother * (sp.Identity(A.shape[0]) - P * A_coarse.I * R * A) * smoother
iteration_matrix = sp.block_collapse(two_grid)
print(iteration_matrix)
print(evaluator.transform(iteration_matrix).symbol().spectral_radius())

jacobi = lfa_lab.jacobi(fine_operator, 1.0)
reference = lfa_lab.coarse_grid_correction(fine_operator, coarse_operator, evaluator.interpolation, evaluator.restriction, coarse_error=None)
print(evaluator.transform(iteration_matrix).symbol().spectral_radius())


