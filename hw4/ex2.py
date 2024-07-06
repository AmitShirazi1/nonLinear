import numpy as np
import matplotlib.pyplot as plt

def generic_grad(f ,gf, t, x0, num_iter=1000):
    x = x0
    fs = [f(x)]
    i = 0
    while i < num_iter:
        x = x - t * gf(x)
        fs.append(f(x))
        i += 1
    return x ,fs

if __name__ == '__main__':
    O = np.array([
        [ 1,  0,  0, -1, -1, -1, -1, -1,  0],
        [ 0,  1, -1, -1, -1, -1, -1, -1, -1],
        [ 1,  1, -1, -1, -1, -1, -1, -1, -1],
        [ 0,  0, -1,  1, -1, -1, -1, -1, -1],
        [-1, -1, -1,  1,  1,  1,  1,  0,  0],
        [-1, -1,  1,  1,  0,  1,  0,  1,  0],
        [-1, -1,  1,  1,  1,  1,  1,  0,  0],
        [-1, -1,  1,  1,  0,  1,  0,  0,  0]
    ])
    N = 8  # number of Knesset members
    step_size = 1/1000
    D = np.array([[np.linalg.norm(O[i] - O[j]) for i in range(N)] for j in range(N)])
    x = np.random.rand(N, 2)
    S = lambda x: 0.5 * np.sum(np.sum(((np.linalg.norm(x[i] - x[j]) ** 2) - (D[i, j] ** 2)) ** 2 for i in range(N)) for j in range(N))
    gS = lambda x: 4 * np.array([np.sum((x[i] - x[j]) * (np.linalg.norm(x[i] - x[j]) ** 2 - D[i, j] ** 2) for j in range(N)) for i in range(N)])

    x, Ss = generic_grad(S, gS, step_size, x)

    colors = ['blue'] * 4 + ['red'] * 4
    plt.scatter(x[:, 0], x[:, 1], c=colors)
    plt.title('Configuration of Knesset members')
    plt.show()

    plt.semilogy(Ss)
    plt.xlabel('Iteration')
    plt.ylabel('log function value')
    plt.title('function value per iteration')
    plt.show()

