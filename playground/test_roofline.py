from evostencils.evaluation.roofline import *

fine_grid_size = (1000, 1000)
operator_stencil_entries = [
    (( 0, -1), -1.0),
    ((-1,  0), -1.0),
    (( 0,  0),  4.0),
    (( 1,  0), -1.0),
    (( 0,  1), -1.0)
]
interpolation_stencil_entries = [
    ((-1, -1), 1.0/4),
    (( 0, -1), 1.0/2),
    (( 1, -1), 1.0/4),
    ((-1,  0), 1.0/2),
    (( 0,  0), 1.0),
    (( 1,  0), 1.0/2),
    ((-1,  1), 1.0/4),
    (( 0,  1), 1.0/2),
    (( 1,  1), 1.0/4),
]

restriction_stencil_entries = [
    ((-1, -1), 1.0/16),
    (( 0, -1), 1.0/8),
    (( 1, -1), 1.0/16),
    ((-1,  0), 1.0/8),
    (( 0,  0), 1.0/4),
    (( 1,  0), 1.0/8),
    ((-1,  1), 1.0/16),
    (( 0,  1), 1.0/8),
    (( 1,  1), 1.0/16),
]

u = base.generate_grid('x', fine_grid_size)
b = base.generate_grid('b', fine_grid_size)
A = base.generate_operator('A', fine_grid_size, stencils.Stencil(operator_stencil_entries))
bytes_per_word = 8
peak_performance = 4 * 16 * 3.3 * 1e9
peak_bandwidth = 34.1 * 1e9
evaluator = RooflineEvaluator(bytes_per_word)


coarsening_factor = 4
u_coarse = multigrid.get_coarse_grid(u, coarsening_factor)
A_coarse = multigrid.get_coarse_operator(A, coarsening_factor)
S_coarse = multigrid.get_coarse_grid_solver(u_coarse)
P = multigrid.get_interpolation(u, u_coarse, stencils.Stencil(interpolation_stencil_entries))
R = multigrid.get_restriction(u, u_coarse, stencils.Stencil(interpolation_stencil_entries))
smoother = base.Inverse(base.Diagonal(A))
tmp = multigrid.correct(smoother, u, A, b)
print(f'Runtime:{evaluator.estimate_runtime(tmp, peak_performance, peak_bandwidth)}')

coarse_grid_correction = base.Multiplication(P, base.Multiplication(S_coarse, R))
tmp = multigrid.correct(coarse_grid_correction, tmp, A, b)
print(f'Runtime:{evaluator.estimate_runtime(tmp, peak_performance, peak_bandwidth)}')
two_grid = multigrid.correct(smoother, tmp, A, b)
print(f'Runtime:{evaluator.estimate_runtime(two_grid, peak_performance, peak_bandwidth)}')

