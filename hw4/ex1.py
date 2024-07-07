import numpy as np
import time
import matplotlib.pyplot as plt

def generic_grad(f ,gf ,lsearch ,x0 ,eps):
    x = x0
    fs, gs, ts = list(), list(), list()
    i = -1
    tic = time.time()
    while True:
        fs.append(f(x))
        g = gf(x)
        gs.append(np.linalg.norm(g))
        ts.append((time.time() - tic) * 1000)
        i += 1
        if (i > 0) and (abs(fs[i-1] - fs[i]) <= eps):
            break
        t = lsearch(f, x ,g)
        x = x - t * g
    return x ,fs ,gs ,ts


def const_step(s):
    def lsearch(f ,xk ,gk):
        return s
    return lsearch

def exact_quad(A):
    try:
        np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise Exception('A is not positive definite') from None
    
    def lsearch(f ,xk ,gk):
        return (np.linalg.norm(gk) ** 2) / (2 * np.matmul(gk.T, np.matmul(A ,gk)))
    return lsearch

def back(alpha ,beta , s):
    def lsearch(f ,xk ,gk):
        t = s
        while f(xk) - f(xk - t * gk) < alpha * t * (np.linalg.norm(gk) ** 2):
            t = beta * t
        return t
    return lsearch


def plot_value_per_iteration(const_val, exact_val, back_val, value_name):
    plt.semilogy(const_val, label='Constant step')
    plt.semilogy(exact_val, label='Exact quadratic')
    plt.semilogy(back_val, label='Backtracking')
    plt.xlabel('Iteration')
    value_name += ' value'
    plt.ylabel('log ' + value_name)
    plt.title(value_name + ' per iteration')
    plt.legend()
    plt.show()

def plot_value_per_time(const_val, exact_val, back_val, value_name, const_time, exact_time, back_time):
    plt.semilogy(const_time, const_val, label='Constant step')
    plt.semilogy(exact_time, exact_val, label='Exact quadratic')
    plt.semilogy(back_time, back_val, label='Backtracking')
    plt.xlabel('Time')
    value_name += ' value'
    plt.ylabel('log ' + value_name)
    plt.title(value_name + ' per time')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    A = np.random.uniform(size=(20,5))
    AT_A = np.matmul(A.T, A)
    f = lambda x: np.dot(x.T, np.dot(AT_A, x))
    gf = lambda x: 2 * np.dot(AT_A, x)
    s = 1 / (2 * np.linalg.eigvals(AT_A).max())
    x0 = np.ones(5)
    eps = 1e-5

    x_const ,fs_const ,gs_const ,ts_const = generic_grad(f ,gf ,const_step(s) ,x0 ,eps)
    x_exact ,fs_exact ,gs_exact ,ts_exact = generic_grad(f ,gf ,exact_quad(AT_A) ,x0 ,eps)
    x_back ,fs_back ,gs_back ,ts_back = generic_grad(f ,gf ,back(0.5 ,0.5 ,1) ,x0 ,eps)

    plot_value_per_iteration(fs_const, fs_exact, fs_back, 'function')
    plot_value_per_iteration(gs_const, gs_exact, gs_back, 'gradient norm')
    plot_value_per_time(gs_const, gs_exact, gs_back, 'gradient norm', ts_const, ts_exact, ts_back)

