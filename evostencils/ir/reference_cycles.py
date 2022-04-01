from evostencils.ir import base, system, smoother
from evostencils.grammar import multigrid as mg


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
    u_c = terminals_coarse_level.approximation
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


# ------------------------------NON LINEAR FAS-------------------------------------------------------
def generate_FAS_v_22_cycle_two_grid(terminals_fine_level: mg.Terminals, rhs: system.RightHandSide):
    u = terminals_fine_level.approximation
    f = rhs
    A = terminals_fine_level.operator
    P = terminals_fine_level.prolongation
    R = terminals_fine_level.restriction
    R_u = R  # TODO: Include operators for restricting solution within Terminals class
    partitioning = terminals_fine_level.red_black_partitioning
    omega = 1
    L = smoother.generate_collective_jacobi(A)  # TODO: Replace with NonLinear Smoothers
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
    A_c = terminals_fine_level.coarse_operator
    # MARK: Modified for FAS
    f1_c = base.Multiplication(R, residual)
    f2_c = base.Multiplication(A_c, base.Multiplication(R_u, u_new))  # Add this term for FAS
    f_c = base.Addition(f1_c, f2_c)
    correction1_c = base.Multiplication(base.CoarseGridSolver(A_c), f_c)
    correction2_c = base.Multiplication(R_u, u_new)  # Subract this term for FAS
    correction_c = base.Subtraction(correction1_c, correction2_c)
    correction = base.Multiplication(P, correction_c)
    cycle = base.Cycle(u_new, f, correction, relaxation_factor=omega)
    u_new = cycle
    # MARK: End modification

    residual = base.Residual(A, u_new, f)
    correction = base.Multiplication(base.Inverse(L), residual)
    cycle = base.Cycle(u_new, f, correction, partitioning=partitioning, relaxation_factor=omega)
    u_new = cycle

    residual = base.Residual(A, u_new, f)
    correction = base.Multiplication(base.Inverse(L), residual)
    cycle = base.Cycle(u_new, f, correction, partitioning=partitioning, relaxation_factor=omega)
    u_new = cycle
    return u_new

def generate_FAS_v_22_cycle_three_grid(terminals_fine_level: mg.Terminals, terminals_coarse_level: mg.Terminals, rhs: system.RightHandSide):
    u = terminals_fine_level.approximation
    f = rhs
    A = terminals_fine_level.operator
    P = terminals_fine_level.prolongation
    R = terminals_fine_level.restriction
    R_u = R  # TODO: Include operators for restricting solution within Terminals class
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
    A_c = terminals_fine_level.coarse_operator
    # MARK: Start: Modified for FAS
    f1_c = base.Multiplication(R, residual)
    f2_c = base.Multiplication(A_c, base.Multiplication(R_u, u_new))  # Add this term for FAS
    f_c = base.Addition(f1_c, f2_c)
    # MARK: End: Modified for FAS
    u_c = terminals_coarse_level.approximation
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
    R_uc = R_c
    residual_c = base.Residual(A_c, u_c_new, f_c)
    A_cc = terminals_coarse_level.coarse_operator
    # MARK: Start: Modified for FAS
    f1_cc = base.Multiplication(R_c, residual_c)
    f2_cc = base.Multiplication(A_cc, base.Multiplication(R_uc, u_c_new))  # Add this term for FAS
    f_cc = base.Addition(f1_cc, f2_cc)
    correction1_cc = base.Multiplication(base.CoarseGridSolver(A_cc), f_cc)
    correction2_cc = base.Multiplication(R_uc, u_c_new)  # Subract this term for FAS
    correction_cc = base.Subtraction(correction1_cc, correction2_cc)
    correction_c = base.Multiplication(P_c, correction_cc)
    # MARK: End: Modified for FAS

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
    # MARK: Start: Modified for FAS
    correction1_c = u_c_new
    correction2_c = base.Multiplication(R_u, u_new)  # Subract this term for FAS
    correction_c = base.Subtraction(correction1_c, correction2_c)
    correction = base.Multiplication(P, correction_c)
    # MARK: End: Modified for FAS
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