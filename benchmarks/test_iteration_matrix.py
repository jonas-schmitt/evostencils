import numpy as np


def generate_gauss_seidel_iteration_matrix(A: np.matrix) -> np.matrix:
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    D = np.matrix(np.diagflat(np.diag(A)))
    return np.linalg.inv(D - L) * U


def generate_jacobi_iteration_matrix(A: np.matrix) -> np.matrix:
    L = -np.tril(A, -1)
    U = -np.triu(A, 1)
    D = np.matrix(np.diagflat(np.diag(A)))
    return np.linalg.inv(D) * (L + U)


def run_iteration(number_of_iterations, G: np.matrix, x_init: np.matrix) -> np.matrix:
    x = x_init
    for _ in range(0, number_of_iterations):
        x = G * x
    return x


def compute_residual(x: np.matrix, A: np.matrix, b: np.matrix) -> np.matrix:
    return b - A*x


def l2_norm(x: np.matrix) -> np.float64:
    return np.linalg.norm(x, 2)
