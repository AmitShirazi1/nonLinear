import numpy as np

def ex1(A, x):
    n = A.shape[0]
    B = np.zeros((n, n))
    # Arrays that represent the indices of the each element of the matrix
    i_indices, j_indices = np.indices((n, n))
    mask = i_indices != j_indices  # Boolean mask to exclude the diagonal
    B[mask] = A.T[mask] + x[j_indices][mask] * (i_indices[mask] + 1)
    return B

A = np.array([
    [11, -7, 4],
    [4, 0, 8],
    [7, 8, 7]
])
x = np.array([10, 11, 13])

B = ex1(A, x)
print(B)
