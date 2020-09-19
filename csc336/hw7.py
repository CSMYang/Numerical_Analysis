import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline

def compute_bound(n):
    to_ret = 1 / np.math.factorial(n)
    w = (np.pi / 2) ** n / 2 ** n
    return to_ret * w - 1e-10


def q3_true_function(x):
    return np.divide(1, np.add(1, 25 * np.square(x)))


if __name__ == '__main__':
    # Q2
    n = 5
    points = np.linspace(0, np.pi / 2, num=n)
    poly = np.polyfit(points, np.sin(points), n - 1)

    points = np.linspace(0, np.pi / 2, num=50)
    results = np.polyval(poly, points)
    true_results = np.sin(points)

    plt.title('Functions evaluated at different points')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.plot(points, results, linewidth=4)
    plt.plot(points, true_results)
    plt.legend(['interpolant', 'sin(x)'])
    plt.show()

    err = abs(np.subtract(true_results, results))
    err_bound = np.ones(50)
    multiplier = (np.pi / 2) ** 5 / (120 * 32)
    err_bound = err_bound * multiplier
    plt.title('Absolute error between the true function and the interpolant')
    plt.xlabel('x')
    plt.ylabel('absolute error')
    plt.plot(points, err)
    plt.plot(points, err_bound)
    plt.legend(['absolute_error', 'err bound'])
    plt.show()

    root = fsolve(compute_bound, 10)
    print("The number of data points needed is: {}".format(root))

    # Q3
    nums = [11, 21]
    for n in nums:
        points = np.linspace(-1, 1, num=n)
        true_results = q3_true_function(points)
        poly = np.polyfit(points, true_results, n-1)
        chebyshev = CubicSpline(points, true_results)

        points = np.linspace(-1, 1, num=101)
        poly_result = np.polyval(poly, points)
        cheby_results = chebyshev(points)
        true_results = q3_true_function(points)
        plt.title('Behavior of each function at n={}'.format(n))
        plt.plot(points, true_results)
        plt.plot(points, cheby_results, '-.')
        plt.plot(points, poly_result, '-.')
        plt.legend(['True function', 'Cubic spline', 'Polynomial'])
        plt.show()