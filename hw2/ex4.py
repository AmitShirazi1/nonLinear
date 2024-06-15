from sys import maxsize
from ex1 import generic_gs
from copy import deepcopy

def gs_denoise_step(mu, a, b, c):
    φ = lambda t: mu * (t - a)**2 + abs(t - b) + abs(t - c)
    eps = 1e-10
    l, u = min(a, b, c) - 1, max(a, b, c) + 1
    return generic_gs(φ, l, u, eps, maxsize)[0]

def gs_denoise(s, alpha, N):
    x = deepcopy(s)
    n = len(s)
    for _ in range(N):
        for k in range(n):
            if k == 0:
                a = s[k]
                b = x[k+1]
                c = x[k+1]
                mu = 2 * alpha
            elif k == n-1:
                a = s[k] 
                b = x[k-1]
                c = x[k-1]
                mu = 2 * alpha
            else:
                a = s[k]
                b = x[k-1]
                c = x[k+1]
                mu = alpha
            x[k] = gs_denoise_step(mu, a, b, c)
    return x


import matplotlib.pyplot as plt
import numpy as np

# plotting the real discrete signal
real_s_1 = [1.]*40
real_s_0 = [0.]*40

plt.plot(range(40), real_s_1, 'black', linewidth=0.7)
plt.plot(range(41, 81), real_s_0, 'black', linewidth=0.7)


# solving the problem
s = np.array([[1.]*40 + [0.]*40]).T + 0.1*np.random.randn(80, 1) # noised signal
x1 = gs_denoise(s, 0.5, 10)
x2 = gs_denoise(s, 0.5, 20)
x3 = gs_denoise(s, 0.5, 30)

plt.plot(range(80), s, 'cyan', linewidth=0.7)
plt.plot(range(80), x1, 'red', linewidth=0.7)
plt.plot(range(80), x2, 'green', linewidth=0.7)
plt.plot(range(80), x3, 'blue', linewidth=0.7)

plt.show()


# t = gs_denoise_step(mu, a, b, c)
# x = gs_denoise(s, alpha, N)