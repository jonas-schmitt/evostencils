from evostencils.expressions import base, multigrid
from evostencils.evaluation import roofline
import evostencils.stencils as stencils

fine_grid_size = (64, 64)
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

coarsening_factor = 4
u_coarse = multigrid.get_coarse_grid(u, coarsening_factor)
A_coarse = multigrid.get_coarse_operator(A, coarsening_factor)
S_coarse = multigrid.get_coarse_grid_solver(u_coarse)
P = multigrid.get_interpolation(u, u_coarse, stencils.Stencil(interpolation_stencil_entries))
R = multigrid.get_restriction(u, u_coarse, stencils.Stencil(interpolation_stencil_entries))
metrics = []
smoother = base.Inverse(base.Diagonal(A))
tmp = multigrid.correct(smoother, u, A, b)
evaluator = roofline.RooflineEvaluator(1, 1, 1)
metrics.extend(evaluator.estimate_correction_performance_metrics(tmp))

coarse_grid_correction = base.Multiplication(P, base.Multiplication(S_coarse, R))
tmp = multigrid.correct(coarse_grid_correction, tmp, A, b)
metrics.extend(evaluator.estimate_correction_performance_metrics(tmp))
two_grid = multigrid.correct(smoother, tmp, A, b)
metrics.extend(evaluator.estimate_correction_performance_metrics(two_grid))
average = evaluator.compute_average_arithmetic_intensity(metrics)
minimum = evaluator.compute_minimum_arithmetic_intensity(metrics)
maximum = evaluator.compute_maximum_arithmetic_intensity(metrics)
print(f'Minimum:{minimum}, Maximum:{maximum}, Average:{average}')

