from evostencils.expressions import base, system, multigrid, transformations
from evostencils.stencils.gallery import *

dimension = 2
min_level = 2
max_level = 10
size = 2**max_level
grid_size = (size, size)
h = 1/(2**max_level)
step_size = (h, h)
coarsening_factor = (2, 2)

grid = base.Grid(grid_size, step_size)
us = base.ZeroApproximation(grid)
vs = base.ZeroApproximation(grid)
f_u = base.RightHandSide('f_u', grid)
f_v = base.RightHandSide('f_v', grid)

problem_name = 'poisson_2D_constant'
stencil_generator = Poisson2D()
interpolation_generator = InterpolationGenerator(coarsening_factor)
restriction_generator = RestrictionGenerator(coarsening_factor)

laplace = base.Operator('Laplace', grid, stencil_generator)
I = base.Identity(grid)
Z = base.ZeroOperator(grid)
Ps = multigrid.Prolongation('multilinear interpolation', grid, multigrid.get_coarse_grid(grid, coarsening_factor), interpolation_generator)
Rs = multigrid.Restriction('full-weighting restriction', grid, multigrid.get_coarse_grid(grid, coarsening_factor), restriction_generator)


A = system.Operator('A', [[laplace, Z], [base.Scaling(-1, I), laplace]])
u = system.Approximation('u', [vs, us])
f = system.RightHandSide('f', [f_v, f_u])
res = multigrid.Residual(A, u, f)
tmp = base.Multiplication(base.Inverse(system.Diagonal(A)), res)

#R = system.Restriction('R', Rs, res.grid)
#P = system.Prolongation('P', Ps, res.grid)
#u_c = system.get_coarse_grid(u.grid, coarsening_factor)
#A_c = system.Operator('A_c', [[multigrid.get_coarse_operator(laplace, u_c[0]), base.ZeroOperator(u_c[1])],
#                              [base.Scaling(-1, base.Identity(u_c[0])), multigrid.get_coarse_operator(laplace, u_c[1])]])
#CGS = multigrid.CoarseGridSolver(A_c)
#tmp = base.Multiplication(R, res)
#tmp = base.Multiplication(CGS, tmp)
#tmp = base.Multiplication(P, tmp)
tmp = multigrid.Cycle(u, f, tmp)
iteration_matrix = transformations.get_system_iteration_matrix(tmp)
foo = 0
