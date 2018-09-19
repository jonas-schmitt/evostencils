from evostencils.expressions import base, multigrid
from evostencils.expressions import transformations
from evostencils.optimizer import Optimizer
import sympy as sp
import lfa_lab as lfa
import evostencils.stencils.constant as constant
import evostencils.stencils.periodic as periodic


from evostencils.evaluation.convergence import *
# Create a 2D grid with step-size (1/32, 1/32).
fine = lfa.Grid(2, [1.0, 1.0])
entries = [
        (( 0, -1), -1.0),
        ((-1,  0), -1.0),
        (( 0,  0),  4.0),
        (( 1,  0), -1.0),
        (( 0,  1), -1.0)
    ]

fine_grid_size = (64,64)

u = base.generate_grid('x', fine_grid_size)
b = base.generate_grid('b', fine_grid_size)
A = base.generate_operator('A', fine_grid_size, constant.Stencil(entries))

# Create a poisson operator.
fine_operator = lfa.gallery.poisson_2d(fine)
coarsening_factor = (2, 2)
u_coarse = multigrid.get_coarse_grid(u, coarsening_factor)
A_coarse = multigrid.get_coarse_operator(A, u_coarse)
P = multigrid.get_interpolation(u, u_coarse)
R = multigrid.get_restriction(u, u_coarse)

coarse_operator = lfa.gallery.poisson_2d(fine.coarse((2,2)))
evaluator = ConvergenceEvaluator(fine_operator, coarse_operator, fine, fine_grid_size, (2, 2))
smoother = base.Inverse(base.Diagonal(A))
tmp = multigrid.correct(A, b, smoother, u, partitioning=base.RedBlackPartitioning, weight=1)
#coarse_grid_correction = base.Multiplication(P, base.Multiplication(multigrid.CoarseGridSolver(u_coarse), R))
#tmp = multigrid.correct(A, b, coarse_grid_correction, tmp, weight=1)
#tmp = multigrid.correct(A, b, smoother, tmp, weight=1)
iteration_matrix = Optimizer.get_iteration_matrix(tmp, u, b)
print(iteration_matrix)
print(evaluator.transform(iteration_matrix).symbol().spectral_radius())



entries = [
        (( 0, -1), -1.0),
        ((-1,  0), -1.0),
        (( 0,  0),  4.0),
        (( 1,  0), -1.0),
        (( 0,  1), -1.0)
    ]

stencil = constant.Stencil(entries)
jacobi = constant.mul(constant.inverse(constant.diagonal(stencil)), constant.add(constant.lower(stencil), constant.upper(stencil)))
jacobi = stencil_to_lfa(jacobi, fine)
#print(jacobi.symbol().spectral_radius())

jacobi = lfa.jacobi(fine_operator, 1)
#print(jacobi.symbol().spectral_radius())
reference = jacobi * lfa.coarse_grid_correction(fine_operator, coarse_operator, evaluator.interpolation, evaluator.restriction, coarse_error=None) * jacobi
#print(reference.symbol().spectral_radius())
rb_jacobi = lfa.rb_jacobi(fine_operator, 1.0)
print(rb_jacobi.symbol().spectral_radius())
block_jacobi = lfa.block_jacobi(fine_operator, (2,2), 1.0)
#print(block_jacobi.symbol().spectral_radius())

rb_block_jacobi = lfa.rb_block_jacobi(fine_operator, (2,2), 1.0)
#print(rb_block_jacobi.symbol().spectral_radius())



