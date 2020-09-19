# CSC 336 Summer 2020 A3Q2 starter code

# Note: you may use the provided code or write your own, it is your choice.

# some general imports
import time
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt


def get_dn(fn):
    """
    Returns dn, given a value for fn, where Ad = b + f (see handout)

    The provided implementation uses the solve_banded approach from A2,
    but feel free to modify it if you want to try to more efficiently obtain
    dn.

    Note, this code uses a global variable for n.
    """

    # the matrix A in banded format
    diagonals = [np.hstack([0, 0, np.ones(n - 2)]),
                 # zeros aren't used, so can be any value.
                 np.hstack([0, -4 * np.ones(n - 2), -2]),
                 np.hstack([9, 6 * np.ones(n - 3), [5, 1]]),
                 np.hstack([-4 * np.ones(n - 2), -2, 0]),
                 # make sure this -2 is in correct spot
                 np.hstack([np.ones(n - 2), 0,
                            0])]  # zeros aren't used, so can be any value.
    A = np.vstack(diagonals) * n ** 3

    b = -(1 / n) * np.ones(n)

    b[-1] += fn

    sol = solve_banded((2, 2), A, b)
    dn = sol[-1]

    return dn


def newton_step(fn):
    """
    This function takes fn as the starting point and computes a newton step
    to get the root.
    """
    # the matrix A in banded format
    diagonals = [np.hstack([0, 0, np.ones(n - 2)]),
                 # zeros aren't used, so can be any value.
                 np.hstack([0, -4 * np.ones(n - 2), -2]),
                 np.hstack([9, 6 * np.ones(n - 3), [5, 1]]),
                 np.hstack([-4 * np.ones(n - 2), -2, 0]),
                 # make sure this -2 is in correct spot
                 np.hstack([np.ones(n - 2), 0,
                            0])]  # zeros aren't used, so can be any value.
    A = np.vstack(diagonals) * n ** 3
    b = np.zeros(n)
    b[-1] = 1
    derivative = solve_banded((2, 2), A, b)
    return np.subtract(fn, np.divide(get_dn(fn), derivative))[-1]


if __name__ == "__main__":
    # experiment code
    brentq_result = []
    fsolve_result = []
    newton_result = []

    brentq_time = []
    fsolve_time = []
    newton_time = []
    ns = []
    for i in range(5, 17):
        n = 2 ** i
        # your code here
        ns.append(n)
        start_time = time.perf_counter()
        fn_b = brentq(get_dn, -10, 10)
        time_used = time.perf_counter() - start_time
        brentq_result.append(fn_b)
        brentq_time.append(time_used)

        start_time = time.perf_counter()
        fn_f = fsolve(get_dn, x0=100)
        time_used = time.perf_counter() - start_time
        fsolve_result.append(fn_f)
        fsolve_time.append(time_used)

        start_time = time.perf_counter()
        fn_n = newton_step(100)
        time_used = time.perf_counter() - start_time
        newton_result.append(fn_n)
        newton_time.append(time_used)

    print('Newton results:')
    print(newton_result)

    print('brentq results:')
    print(brentq_result)

    print('fsolve results:')
    print(fsolve_result)

    plt.title("Root vs Size of matrix")
    plt.xlabel('Size of matrix')
    plt.ylabel('Value of the root')
    plt.plot(ns, newton_result, '*-r')
    plt.plot(ns, brentq_result, '*-g')
    plt.plot(ns, fsolve_result, '*-b')
    plt.legend(['Newton', 'Brentq', 'fsolve'])
    plt.show()

    plt.title("Time taken vs Size of matrix")
    plt.xlabel('Size of matrix')
    plt.ylabel('Time taken')
    plt.plot(ns, newton_time, '*-r')
    plt.plot(ns, brentq_time, '*-g')
    plt.plot(ns, fsolve_time, '*-b')
    plt.legend(['Newton', 'Brentq', 'fsolve'])
    plt.show()
