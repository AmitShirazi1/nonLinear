import numpy as np
from ex1 import generic_grad, const_step
import matplotlib.pyplot as plt

def noise_grad(f ,gf, lsearch, x0, mu, sigma ,epsilon):
    x = x0
    n = len(x)
    fs = list()
    i = -1
    while True:
        fs.append(f(x) + 0.25)
        g = gf(x)
        i += 1
        if (i > 0) and (abs(fs[i-1] - fs[i]) <= epsilon):
            break
        t = lsearch(f, x ,g)
        x = x - t * g + np.random.normal(mu, sigma, n)
    return x ,fs

def ex3(mu, sigma, x0, epsilon):
    f = lambda x: (x[0] ** 2) + (x[1] ** 4) - (x[1] ** 2)
    gf = lambda x: np.array([2 * x[0], 4 * (x[1] ** 3) - 2 * x[1]])
    lsearch = const_step(0.1)

    x, fs, _, _ = generic_grad(f ,gf ,lsearch ,x0 ,epsilon)
    fs = [val + 0.25 for val in fs]
    x_noise, fs_noise = noise_grad(f, gf, lsearch, x0, mu, sigma, epsilon)

    return x, x_noise, fs, fs_noise


if __name__ == '__main__':
    x, x_noise, fs, fs_noise = ex3(0, 0.0005, [100, 0], 1e-8)
    print('x:\n' + str(x))
    print('x_noise:\n' + str(x_noise))
    print('fs:\n' + str(fs))
    print('fs noise:\n' + str(fs_noise))

    plt.semilogy(fs, label='fs')
    plt.semilogy(fs_noise, label='fs noise')
    plt.xlabel('Iteration')
    plt.ylabel('Log function value')
    plt.title('Function value per iteration')
    plt.legend()
    plt.show()