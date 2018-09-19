from evostencils.evaluation.roofline import *
import evostencils.stencils.constant as constant

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
A = base.generate_operator('A', fine_grid_size, constant.Stencil(operator_stencil_entries))
bytes_per_word = 8
peak_performance = 4 * 16 * 3.3 * 1e9
peak_bandwidth = 34.1 * 1e9
evaluator = RooflineEvaluator(peak_performance, peak_bandwidth, bytes_per_word)


coarsening_factor = (2, 2)
u_coarse = multigrid.get_coarse_grid(u, coarsening_factor)
A_coarse = multigrid.get_coarse_operator(A, u_coarse)
S_coarse = multigrid.get_coarse_grid_solver(u_coarse)
P = multigrid.get_interpolation(u, u_coarse, constant.Stencil(interpolation_stencil_entries))
R = multigrid.get_restriction(u, u_coarse, constant.Stencil(interpolation_stencil_entries))
smoother = base.Inverse(base.Diagonal(A))
tmp = multigrid.correct(A, b, smoother, u, weight=1)
print(f'Runtime:{evaluator.estimate_runtime(tmp)}')

coarse_grid_correction = base.Multiplication(P, base.Multiplication(S_coarse, R))
tmp = multigrid.correct(A, b, coarse_grid_correction, tmp, weight=1)
print(f'Runtime:{evaluator.estimate_runtime(tmp)}')
two_grid = multigrid.correct(A, b, smoother, tmp, weight=1)
print(f'Runtime:{evaluator.estimate_runtime(two_grid)}')

