from evostencils.expressions import base, system, smoother
from evostencils.initialization import multigrid as mg


def generate_v_22_cycle_three_grid(terminals_fine_level: mg.Terminals, terminals_coarse_level: mg.Terminals, rhs: system.RightHandSide):
    u = terminals_fine_level.approximation
    f = rhs
    A = terminals_fine_level.operator
    P = terminals_fine_level.prolongation
    R = terminals_fine_level.restriction
    partitioning = terminals_fine_level.red_black_partitioning
    omega = 1.0
    L = smoother.generate_collective_jacobi(A)
    # Pre-Smoothing step 1
    residual = base.Residual(A, u, f)
    correction = base.Multiplication(base.Inverse(L), residual)
    cycle = base.Cycle(u, f, correction, partitioning=partitioning, relaxation_factor=omega)
    u_new = cycle

    # Pre-Smoothing step 2
    residual = base.Residual(A, u_new, f)
    correction = base.Multiplication(base.Inverse(L), residual)
    cycle = base.Cycle(u_new, f, correction, partitioning=partitioning, relaxation_factor=omega)
    u_new = cycle

    # Solve residual equation in coarse grid
    residual = base.Residual(A, u_new, f)
    f_c = base.Multiplication(R, residual)
    u_c = system.ZeroApproximation(terminals_fine_level.coarse_grid)
    A_c = terminals_fine_level.coarse_operator
    L_c = smoother.generate_collective_jacobi(A_c)

    # Pre-Smoothing step 1
    residual_c = base.Residual(A_c, u_c, f_c)
    correction_c = base.Multiplication(base.Inverse(L_c), residual_c)
    cycle_c = base.Cycle(u_c, f_c, correction_c, relaxation_factor=omega, predecessor=cycle)
    u_c_new = cycle_c

    # Pre-Smoothing step 2
    residual_c = base.Residual(A_c, u_c_new, f_c)
    correction_c = base.Multiplication(base.Inverse(L_c), residual_c)
    cycle_c = base.Cycle(u_c_new, f_c, correction_c, partitioning=partitioning, relaxation_factor=omega, predecessor=cycle)
    u_c_new = cycle_c

    # Solve residual equation in coarse grid
    R_c = terminals_coarse_level.restriction
    P_c = terminals_coarse_level.prolongation
    residual_c = base.Residual(A_c, u_c_new, f_c)
    f_cc = base.Multiplication(R_c, residual_c)
    A_cc = terminals_coarse_level.coarse_operator
    correction_cc = base.Multiplication(base.CoarseGridSolver(A_cc), f_cc)
    correction_c = base.Multiplication(P_c, correction_cc)
    # Coarse grid correction
    cycle_c = base.Cycle(u_c_new, f_c, correction_c, relaxation_factor=omega, predecessor=cycle)
    u_c_new = cycle_c

    # Post-Smoothing step 1
    residual_c = base.Residual(A_c, u_c_new, f_c)
    correction_c = base.Multiplication(base.Inverse(L_c), residual_c)
    cycle_c = base.Cycle(u_c_new, f_c, correction_c, partitioning=partitioning, relaxation_factor=omega, predecessor=cycle)
    u_c_new = cycle_c

    # Post-Smoothing step 2
    residual_c = base.Residual(A_c, u_c_new, f_c)
    correction_c = base.Multiplication(base.Inverse(L_c), residual_c)
    cycle_c = base.Cycle(u_c_new, f_c, correction_c, partitioning=partitioning, relaxation_factor=omega, predecessor=cycle)
    u_c_new = cycle_c

    # Coarse grid correction
    correction = base.Multiplication(P, u_c_new)
    cycle = base.Cycle(u_new, f, correction, relaxation_factor=omega)
    u_new = cycle

    # Post-Smoothing step 1
    residual = base.Residual(A, u_new, f)
    correction = base.Multiplication(base.Inverse(L), residual)
    cycle = base.Cycle(u_new, f, correction, partitioning=partitioning, relaxation_factor=omega)
    u_new = cycle

    # Post-Smoothing step 2
    residual = base.Residual(A, u_new, f)
    correction = base.Multiplication(base.Inverse(L), residual)
    cycle = base.Cycle(u_new, f, correction, partitioning=partitioning, relaxation_factor=omega)
    u_new = cycle
    return u_new


def generate_v_22_cycle_two_grid(terminals_fine_level: mg.Terminals, rhs: system.RightHandSide):
    u = terminals_fine_level.approximation
    f = rhs
    A = terminals_fine_level.operator
    P = terminals_fine_level.prolongation
    R = terminals_fine_level.restriction
    partitioning = terminals_fine_level.red_black_partitioning
    omega = 1
    L = smoother.generate_collective_jacobi(A)
    # L = smoother.generate_collective_block_jacobi(A, ((2, 2), (2, 2)))
    # L = smoother.generate_decoupled_jacobi(A)

    residual = base.Residual(A, u, f)
    correction = base.Multiplication(base.Inverse(L), residual)
    cycle = base.Cycle(u, f, correction, partitioning=partitioning, relaxation_factor=omega)
    u_new = cycle

    residual = base.Residual(A, u_new, f)
    correction = base.Multiplication(base.Inverse(L), residual)
    cycle = base.Cycle(u_new, f, correction, partitioning=partitioning, relaxation_factor=omega)
    u_new = cycle

    residual = base.Residual(A, u_new, f)
    f_c = base.Multiplication(R, residual)
    A_c = terminals_fine_level.coarse_operator
    correction_c = base.Multiplication(base.CoarseGridSolver(A_c), f_c)
    correction = base.Multiplication(P, correction_c)
    cycle = base.Cycle(u_new, f, correction, relaxation_factor=omega)
    u_new = cycle

    residual = base.Residual(A, u_new, f)
    correction = base.Multiplication(base.Inverse(L), residual)
    cycle = base.Cycle(u_new, f, correction, partitioning=partitioning, relaxation_factor=omega)
    u_new = cycle

    residual = base.Residual(A, u_new, f)
    correction = base.Multiplication(base.Inverse(L), residual)
    cycle = base.Cycle(u_new, f, correction, partitioning=partitioning, relaxation_factor=omega)
    u_new = cycle
    return u_new






