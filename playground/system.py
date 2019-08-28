from evostencils.expressions import base, system, multigrid, transformations, partitioning
from evostencils.stencils.gallery import *
import lfa_lab
from evostencils.evaluation.convergence import ConvergenceEvaluator

dimension = 2
min_level = 2
max_level = 10
size = 2**max_level
grid_size = (size, size)
h = 1/(2**max_level)
step_size = (h, h)
coarsening_factor = (2, 2)
coarsening_factors = [coarsening_factor, coarsening_factor]
grid = base.Grid(grid_size, step_size)
us = base.ZeroApproximation(grid)
vs = base.ZeroApproximation(grid)
f_u = base.RightHandSide('f_u', grid)
f_v = base.RightHandSide('f_v', grid)

problem_name = 'poisson_2D_constant'
stencil_generator = Poisson2D()
interpolation_generator = MultilinearInterpolationGenerator(coarsening_factor)
restriction_generator = FullWeightingRestrictionGenerator(coarsening_factor)

laplace = base.Operator('Laplace', grid, stencil_generator)
I = base.Identity(grid)
Z = base.ZeroOperator(grid)


A = system.Operator('A', [[laplace, I], [Z, laplace]])
u = system.Approximation('u', [vs, us])
f = system.RightHandSide('f', [f_v, f_u])
res = multigrid.Residual(A, u, f)
tmp = base.Multiplication(base.Inverse(system.ElementwiseDiagonal(A)), res)
u_new = multigrid.Cycle(u, f, tmp, weight=0.6, partitioning=partitioning.Single)
convergence_evaluator = ConvergenceEvaluator([lfa_lab.Grid(dimension, step_size), lfa_lab.Grid(dimension, step_size)], coarsening_factors, dimension)
lfa_node = convergence_evaluator.transform(u_new)
convergence_factor = convergence_evaluator.compute_spectral_radius(u_new)
print(f'Spectral radius: {convergence_factor}')
