def generic_bisect(f, df, l, u, eps, k):
    fv = []
    fv.append(f(u))

    while u-l> eps and k>0:
        x = (l+u)/2
        k -= 1
        if df(x) < 0:
            l=x
        else: 
            u=x
        fv.append(f(x))     
    return x, fv


def generic_newton(f, df, ddf, x0, k):
    fv = []
    fv.append(f(x0))
    x = x0
    while k > 0:
        x = x - df(x)/ddf(x)
        k -= 1
        fv.append(f(x))
    return x, fv


def generic_hybrid(f, df, ddf, l, u, x0, eps, k):
    fv = []
    fv.append(f(u))
    x = x0
    while abs(u - l) > eps and k > 0 and abs(df(x)) > eps:
        x_newton = x - df(x) / ddf(x)
        k -= 1
        if (l < x_newton < u) and abs(df(x_newton)) < (0.99 * abs(df(x))):   
            x = x_newton
        else: 
            x = (l + u) / 2
        if (df(x) * df(u)) > 0:
            u = x
        else:
            l = x
        fv.append(f(x))     
    return x, fv


def generic_gs(f, l, u, eps, k):
    # Golden section search
    c = 0.61803398875
    fv = []
    fv.append(f(u))
    x2 = c*l + (1-c)*u
    f2 = f(x2)
    x3 = (1-c)*l + c*u
    f3 = f(x3)

    while abs(u - l) > eps and k > 0:
        if f2 < f3:
            u = x3
            x3 = x2
            f3 = f2
            x2 = c*l + (1-c)*u
            f2 = f(x2)
        else:
            l = x2
            x2 = x3
            f2 = f3
            x3 = (1-c)*l + c*u
            f3 = f(x3)
        k -= 1
        fv.append(f(u))
    return l, fv


f = lambda x: (x - 1)**3 + 1/(1 - x**2)
df = lambda x: 3*(x - 1)**2 + 2*x/(1 - x**2)**2
ddf = lambda x: 6*(x - 1) + (2 + 6*x**2)/(1 - x**2)**3
l, u = -0.999, 0.9
k = 50
eps = 1e-6
x0 = u

x1, fv1 = generic_bisect(f, df, l, u, eps, k)
x2, fv2 = generic_newton(f, df, ddf, x0, k)
x3, fv3 = generic_hybrid(f, df, ddf, l, u, x0, eps, k)
x4, fv4 = generic_gs(f, l, u, eps, k)


import matplotlib.pyplot as plt

# Adding to each value in fv
def normalize_and_semilogy(method, fv, color):
    fv_normalized = [value + 3.08891254695156 for value in fv]
    plt.semilogy(fv_normalized, label=method, color='C' + str(color))

# Plotting each fv_normalized with semilogy in different colors
color = 0
for method, fv in zip(['bisect', 'newton', 'hybrid', 'gs'], [fv1, fv2, fv3, fv4]):
    normalize_and_semilogy(method, fv, color)
    color += 1

plt.xlabel('Iteration')
plt.ylabel('Logarithmic span of fv')
plt.title('Semilogarithmic plot of fv (normalized)')
plt.legend()  # To add a legend to distinguish the different lines
plt.show()


if __name__ == '__main__':
    p = lambda x: - 3.55*(x**3) + 1.1*(x**2) + 0.765*x - 0.74
    dp = lambda x: - 10.65*(x**2) + 2.2*x + 0.765
    l, u = -1, 0.5554
    k = 50
    eps = 1e-5
    x0 = u

    x, fv = generic_hybrid(p, p, dp, l, u, x0, eps, k)
    print(x)