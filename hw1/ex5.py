import numpy as np
import matplotlib.pyplot as plt

X = np.array([[-0.966175231649752, -0.920529100440521, -0.871040946427231, -0.792416754493313, -0.731997794083466, -0.707678784846507, -0.594776425699584, -0.542182374657374, -0.477652051223985, -0.414002394497506, -0.326351540865686, -0.301458382421319, -0.143486910424499, -0.0878464728184052, -0.0350835941699658, 0.0334396260398352, 0.0795033683251447, 0.202974351567305, 0.237382785959596, 0.288908922672592, 0.419851917880386, 0.441532730387388, 0.499570508388721, 0.577394288619662, 0.629734626483965, 0.690534081997171, 0.868883439039411, 0.911733893303862, 0.940260537535768, 0.962286449219438],[1.61070071922315, 2.00134259950511, 2.52365719332252, 2.33863055618848, 2.46787274461421, 2.92596278963705, 4.49457749339454, 5.01302648557115, 5.53887922607839, 5.59614305167494, 5.3790027966219, 4.96873291187938, 3.56249278950514, 2.31744895283007, 2.39921966442751, 1.52592143155874, 1.42166345066052, 1.19058953217964, 1.23598301133586, 0.461229833080578, 0.940922128674924, 0.73146046340835, 0.444386541739061, 0.332335616103906, 0.285195114684272, 0.219953363135822, 0.234575259776606, 0.228396325882262, 0.451944920264431, 0.655793276158532]])

def fit_rational(X):
    X = X.T
    ones_col = np.ones(X.shape[0])
    x_col = X[:,0]  # X[:,0] is the first column of X
    x_squared_col =np.power(x_col, 2)
    y_col = X[:,1]  # X[:,1] is the second column of X
    minus_yx_col = - np.multiply(y_col, x_col)
    minus_yx_squared_col = - np.multiply(y_col, x_squared_col)
    A = np.column_stack((ones_col, x_col, x_squared_col, minus_yx_col, minus_yx_squared_col))
    
    u = np.linalg.lstsq(A, y_col, rcond=None)[0]
    return u

def plot_rational(u, X):
    if u.shape[0] == 5:
        f = lambda x: (u[0] + u[1]*x + u[2]*x**2)/(1 + u[3]*x + u[4]*x**2)
    elif u.shape[0] == 6:
        f = lambda x: (u[0] + u[1]*x + u[2]*x**2)/(u[3] + u[4]*x + u[5]*x**2)
    X = X.T
    x_col, y_col = X[:,0], X[:,1]
    x_space = np.linspace(-1, 1, 100)
    y = f(x_space)
    plt.plot(x_space, y, 'r', label='approximation', linewidth=2)
    plt.plot(x_col, y_col, 'bo', label='data points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Rational approximation')
    plt.legend()
    plt.show()

u = fit_rational(X)
print('u:', u)
plot_rational(u, X)


def fit_rational_normed(X):
    X = X.T
    minus_ones_col = - np.ones(X.shape[0])
    x_col = X[:,0]  # X[:,0] is the first column of X
    x_squared_col = np.power(x_col, 2)
    y_col = X[:,1]  # X[:,1] is the second column of X
    yx_col = np.multiply(y_col, x_col)
    yx_squared_col = np.multiply(y_col, x_squared_col)
    A = np.column_stack((minus_ones_col, - x_col, - x_squared_col, y_col, yx_col, yx_squared_col))
    ATA = np.dot(A.T, A)
    U, S, V = np.linalg.svd(ATA)
    u = U[:, -1]
    return u

u = fit_rational_normed(X)
print('u:', u)
plot_rational(u, X)
