import numpy as np
from scipy import sparse

# Generate the matrix of the one dimensional discrete Poisson equation on N grid points
# N: grid points 
# The matrix is returned in CRS format
def generate_1D_matrix(N):
    assert(N > 1),"N > 1 required"
    a = sparse.diags([2] * N, 0, format="csr");
    b = sparse.diags([-1] * (N-1), 1, format="csr");
    c = sparse.diags([-1] * (N-1), -1, format="csr");
    A = a + b + c;
    # To convert to a numpy array
    # A = A.toarray();
    return A;

# Generate the matrix of the two dimensional discrete Poisson equation on a square grid
# N + 2: dimension of the grid 
# The matrix is returned in CRS format
def generate_2D_matrix(N):
    assert(N > 1),"N > 1 required"
    B = generate_1D_matrix(N);
    I = sparse.eye(N);
    A = sparse.kron(B, I) + sparse.kron(I, B);
    # To convert to a numpy array
    # A = A.toarray();
    return A;

# Generate the matrix of the three dimensional discrete Poisson on a cubic grid
# N + 2: Dimension of the grid
# THe matrix is returned in CRS format

def generate_3D_matrix(N):
    assert(N > 1),"N > 1 required"
    A = 6 * sparse.eye(N**3, format="lil");
    off_i = N**2;
    off_j = N;
    off_k = 1; 
    for i in range(0, N):
        for j in range(0, N):
            for k in range(0, N):
                base = i*(N**2) + j*N + k;  
                if i != 0:
                    A[base, base - off_i] = -1;
                if j != 0:
                    A[base, base - off_j] = -1;
                if k != 0:
                    A[base, base - off_k] = -1;
                if i != N-1:
                    A[base, base + off_i] = -1;
                if j != N-1:
                    A[base, base + off_j] = -1;
                if k != N-1:
                    A[base, base + off_k] = -1;
    return sparse.csr_matrix(A);
