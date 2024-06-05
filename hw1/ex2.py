import numpy as np

def ex2(A, B, n, b):
    m = A.shape[0]
    P = np.zeros((n*m, n*m))
    for i in range(n):
        for j in range(n):
            # Enter A matrix as blocks in the diagonal of P
            if i == j:
                P[i*m:(i+1)*m, j*m:(j+1)*m] = A

            # Enter B matrix as blocks near the diagonal of P
            elif j == i + 1:
                P[i*m:(i+1)*m, j*m:(j+1)*m] = B.T
            elif i == j + 1:
                P[i*m:(i+1)*m, j*m:(j+1)*m] = B

            else:
                np.zeros((m, m))

    y = np.kron(np.arange(1, n + 1), b)
    z = np.kron(np.ones(m), y)
    Q = np.kron(A, P)

    x = np.linalg.solve(Q, z)
    return x


A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = A + 2 * np.ones((3, 3))
b = np.array([1, 2, 3])
n = 4
print(ex2(A, B, n, b))
