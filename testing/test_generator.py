from evostencils.expressions import base, multigrid, partitioning, transformations
from evostencils.stencils import gallery
from evostencils.exastencils.generation import *
from evostencils.evaluation.convergence import ConvergenceEvaluator, lfa_sparse_stencil_to_constant_stencil, stencil_to_lfa
from evostencils.evaluation.roofline import RooflineEvaluator
import lfa_lab as lfa

dimension = 2
size = 2**10
grid_size = (size, size)
step_size = (0.00390625, 0.00390625)
coarsening_factor = (2, 2)

lfa_grid = lfa.Grid(dimension, step_size)
lfa_operator = lfa.gallery.poisson_2d(lfa_grid)
lfa_coarse_operator = lfa.gallery.poisson_2d(lfa_grid.coarse(coarsening_factor))
convergence_evaluator = ConvergenceEvaluator(lfa_grid, coarsening_factor, dimension, lfa.gallery.ml_interpolation, lfa.gallery.fw_restriction)
roofline_evaluator = RooflineEvaluator()

lfa_interpolation = lfa.gallery.ml_interpolation_stencil(lfa_grid, lfa_grid.coarse(coarsening_factor))
interpolation = lfa_sparse_stencil_to_constant_stencil(lfa_interpolation)


def generate_interpolation(_):
    return interpolation


lfa_restriction = lfa.gallery.fw_restriction_stencil(lfa_grid, lfa_grid.coarse(coarsening_factor))
restriction = lfa_sparse_stencil_to_constant_stencil(lfa_restriction)


def generate_restriction(_):
    return restriction


u = base.generate_grid('u', grid_size, step_size)
b = base.generate_rhs('f', grid_size, step_size)
A = base.generate_operator_on_grid('A', u, gallery.generate_poisson_2d)

u_coarse = multigrid.get_coarse_grid(u, coarsening_factor)
A_coarse = multigrid.get_coarse_operator(A, u_coarse)
P = multigrid.get_interpolation(u, u_coarse, generate_interpolation)
R = multigrid.get_restriction(u, u_coarse, generate_restriction)
generator = ProgramGenerator('2D_FD_Poisson', '/local/ja42rica/ScalaExaStencil', A, u, b, dimension, coarsening_factor, P, R)
# Jacobi
smoother = base.Inverse(base.Diagonal(A))
correction = base.mul(smoother, multigrid.residual(A, u, b))
jacobi = multigrid.cycle(u, b, correction, partitioning=partitioning.Single, weight=1)
#print("Generating Jacobi\n")

# Block-Jacobi
smoother = base.Inverse(base.BlockDiagonal(A, (2, 2)))
correction = base.mul(smoother, multigrid.residual(A, u, b))
block_jacobi = multigrid.cycle(u, b, correction, partitioning=partitioning.Single, weight=1)

# Red-Black-Jacobi
smoother = base.Inverse(base.Diagonal(A))
correction = base.mul(smoother, multigrid.residual(A, u, b))
rb_jacobi = multigrid.cycle(u, b, correction, partitioning=partitioning.RedBlack, weight=0.8)
"""
# Two-Grid
tmp = rb_jacobi
tmp = multigrid.residual(A, tmp, b)
tmp = base.mul(multigrid.get_restriction(u, u_coarse, generate_restriction), tmp)
tmp = base.mul(multigrid.CoarseGridSolver(A_coarse), tmp)
tmp = base.mul(multigrid.get_interpolation(u, u_coarse, generate_interpolation), tmp)
tmp = multigrid.cycle(rb_jacobi, b, tmp)
tmp = multigrid.cycle(tmp, b, base.mul(base.Inverse(base.Diagonal(A)), mg.residual(A, tmp, b)), weight=0.8, partitioning=partitioning.RedBlack)

iteration_matrix = transformations.get_iteration_matrix(tmp)
print(iteration_matrix)
storages = generator.generate_storage(8)
program = generator.generate_boilerplate(storages, 8)
program += generator.generate_cycle_function(tmp, storages)
print(program)
generator.write_program_to_file(program)

zero = base.ZeroGrid(u_coarse.size, u_coarse.step_size)
# Three-Grid
tmp = mg.cycle(u, b, base.mul(mg.get_restriction(u, u_coarse, generate_restriction), mg.residual(A, u, b)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(zero, tmp.correction, mg.residual(A_coarse, zero, tmp.correction)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(tmp.correction.iterate, tmp.correction.rhs, base.mul(base.Inverse(base.Diagonal(A_coarse)), tmp.correction.correction)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(tmp.correction, tmp.correction.rhs, mg.residual(A_coarse, tmp.correction, tmp.correction.rhs)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(tmp.correction.iterate, tmp.correction.rhs, base.mul(base.Inverse(base.Diagonal(A_coarse)), tmp.correction.correction)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(tmp.correction, tmp.correction.rhs, mg.residual(A_coarse, tmp.correction, tmp.correction.rhs)))
u_coarse_coarse = multigrid.get_coarse_grid(u_coarse, coarsening_factor)
A_coarse_coarse = multigrid.get_coarse_operator(A_coarse, u_coarse_coarse)
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(tmp.correction.iterate, tmp.correction.rhs, base.mul(mg.get_restriction(u_coarse, u_coarse_coarse, generate_restriction), tmp.correction.correction)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(tmp.correction.iterate, tmp.correction.rhs, base.mul(mg.get_coarse_grid_solver(A_coarse_coarse), tmp.correction.correction)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, mg.cycle(tmp.correction.iterate, tmp.correction.rhs, base.mul(mg.get_interpolation(u_coarse, u_coarse_coarse, generate_interpolation), tmp.correction.correction)))
tmp = mg.cycle(tmp.iterate, tmp.rhs, base.mul(mg.get_interpolation(u, u_coarse, generate_interpolation), tmp.correction))
"""

# Two-Grid
tmp = rb_jacobi
tmp = multigrid.residual(A, tmp, b)
tmp = base.mul(multigrid.get_restriction(u, u_coarse, generate_restriction), tmp)
tmp = base.mul(multigrid.CoarseGridSolver(A_coarse), tmp)
tmp = base.mul(multigrid.get_interpolation(u, u_coarse, generate_interpolation), tmp)
tmp = base.mul(base.Inverse(base.Diagonal(A)), tmp)
tmp = base.mul(base.Inverse(base.Diagonal(A)), tmp)
tmp = base.mul(base.Inverse(base.Diagonal(A)), tmp)
tmp = multigrid.cycle(rb_jacobi, b, tmp)
tmp = multigrid.cycle(tmp, b, base.mul(base.Inverse(base.Diagonal(A)), mg.residual(A, tmp, b)), weight=0.8, partitioning=partitioning.RedBlack)

iteration_matrix = transformations.get_iteration_matrix(tmp)
print(iteration_matrix)
print(convergence_evaluator.compute_spectral_radius(iteration_matrix))
#new_iteration_matrix = transformations.simplify_iteration_matrix(iteration_matrix)
#transformations.simplify_iteration_matrix_on_all_levels(new_iteration_matrix)
#print(new_iteration_matrix)
#print(convergence_evaluator.compute_spectral_radius(new_iteration_matrix))

#weights = transformations.obtain_weights(tmp)
#for i in range(len(weights)):
#    weights[i] = 0.8
#tail = transformations.set_weights(tmp, weights)
#print("Generating Multigrid\n")
storages = generator.generate_storage(8)
program = generator.generate_boilerplate(storages, 8)
program += generator.generate_cycle_function(tmp, storages)
print(program)
#generator.write_program_to_file(program)
#print(generator.execute())
#print_declarations(temporaries)
