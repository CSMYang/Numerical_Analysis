# CSC336 Summer 2020 HW6 starter code

# some basic imports
import numpy as np
import numpy.linalg as LA
from scipy.linalg import solve
import matplotlib.pyplot as plt
import time


# Question 1 [autotested]

# Complete the following 5 functions that compute the Jacobians
# of the 5 nonlinear systems in Heath Exercise 5.9

# NOTE: For this question, assume x.shape == (2,),
#      so you can access components of
#      x as x[0] and x[1].

def jac_a(x):
    return np.array([[2 * x[0], 2 * x[1]],
                     [2 * x[0], -1]])


def jac_b(x):
    return np.array([[2 * x[0] + x[1] ** 3, 3 * x[0] * np.square(x[1])],
                     [6 * x[0] * x[1],
                      3 * np.square(x[0]) - 3 * np.square(x[1])]])


def jac_c(x):
    return np.array([[1 - 2 * x[1], 1 - 2 * x[0]],
                     [2 * x[0] - 2, 2 * x[1] + 2]])


def jac_d(x):
    return np.array([[3 * np.square(x[0]), -2 * x[1]],
                     [1 + 2 * x[0] * x[1], np.square(x[0])]])


def jac_e(x):
    return np.array([[2 * np.cos(x[0]) - 5, - np.sin(x[1])],
                     [-4 * np.sin(x[0]), 2 * np.cos(x[1]) - 5]])


########################################
# Question 2

# NOTE: You may use the provided code below or write your own for Q2,
# it is up to you.

# NOTE: For this question, if you use the provided code, it assumes
#      that x.shape == (3,1),
#      so you need to access components of
#      x as x[0,0], x[1,0], and x[2,0]


# useful for checking convergence behaviour of the fixed point method
from scipy.linalg import eigvals


def spectral_radius(A):
    return np.max(np.abs(eigvals(A)))


# This is essentially the same as estimate_convergence_rate from HW5,
# but for non-scalar x values.
def conv_info(xs):
    """
    Returns approximate values of the convergence rate r,
    constant C, and array of error estimates, for the given
    sequence, xs, of x values.

    Note: xs should be an np.array with shape k x n,
    where n is the dimension of each x
    and k is the number of elements in the sequence.

    This code uses the infinity norm for the norm of the errors.
    """

    errs = []
    nxs = np.diff(np.array(xs), axis=0)

    for row in nxs:
        errs.append(LA.norm(row, np.inf))

    r = np.log(errs[-1] / errs[-2]) / np.log(errs[-2] / errs[-3])
    c = errs[-1] / (errs[-2] ** r)

    return r, c, errs


# functions for doing root finding
def fixed_point(g, x0, atol=1e-14):
    """
    Simple implementation of a fixed point iteration for systems,
    with stopping criterion based on infinity norm of difference between
    values of x on consecutive iterations.

    The fixed point iteration is x_{k+1} = g(x_k).

    Returns the sequence of x values as an np.array with shape k x n,
    where n is the dimension of each x
    and k is the number of elements in the sequence.
    """
    x = x0
    xs = [x0]
    err = 1
    while err > atol:
        x = g(x)
        xs.append(x)
        err = LA.norm(xs[-2] - xs[-1], np.inf)
    return np.array(xs)


def newton(f, J, x0, atol=1e-14):
    """
    Simple implementation of Newton's method for systems,
    with stopping criterion based on infinity norm of difference between
    values of x on consecutive iterations.

    f is the function and J computes the Jacobian.

    Returns the sequence of x values as an np.array with shape k x n,
    where n is the dimension of each x
    and k is the number of elements in the sequence.
    """
    x = x0
    xs = [x0]
    err = 1
    while err > atol:
        x = x - solve(J(x), f(x))
        xs.append(x)
        err = LA.norm(xs[-2] - xs[-1], np.inf)
    return np.array(xs)


def broyden(f, x0, atol=1e-14):
    """
    Simple implementation of Broyden's method for systems,
    with stopping criterion based on infinity norm of difference between
    values of x on consecutive iterations.

    f is the function to find the root for.

    Initially, the approximate Jacobian is set to the identity matrix.

    Returns the sequence of x values as an np.array with shape k x n,
    where n is the dimension of each x
    and k is the number of elements in the sequence.
    """
    B = np.identity(len(x0))
    x = x0
    xs = [x0]
    err = 1
    fx = f(x)
    while (err > atol):
        s = solve(B, -fx)
        x = x + s
        xs.append(x)
        next_fx = f(x)
        y = next_fx - fx
        # update approximate Jacobian
        B = B + ((y - B @ s) @ s.T) / (s.T @ s)
        fx = next_fx
        err = LA.norm(xs[-2] - xs[-1], np.inf)
    return np.array(xs)


# the function g from the question description,
# (i.e. x = g(x) is the fixed-point iteration)
def g(x):
    cos, sin = np.cos(x), np.sin(x)
    x_1 = np.subtract(np.divide(np.square(x[1]), 9) + np.divide(sin[2], 3),
                      np.divide(cos[0], 81))
    x_2 = np.divide(sin[0], 3) + np.divide(cos[2], 3)
    x_3 = np.subtract(np.divide(x[1], 3) + np.divide(sin[2], 6),
                      np.divide(cos[0], 9))
    return np.array([x_1, x_2, x_3])


# computes the jacobian of g(x)
def getG(x):
    cos, sin = np.cos(x), np.sin(x)
    row_0 = np.array([np.divide(sin[0, 0], 81),
                      np.divide(2 * x[1, 0], 9), np.divide(cos[2, 0], 3)])
    row_1 = np.array([np.divide(cos[0, 0], 3), 0, -np.divide(sin[2, 0], 3)])
    row_2 = np.array([np.divide(sin[0, 0], 9), 1 / 3, np.divide(cos[2, 0], 6)])
    return np.array([row_0, row_1, row_2])


# x = g(x) rewritten in the form f(x), pass this into broyden / newton code
def f(x):
    return x - g(x)  # pass #your code here


# computes the jacobian of f(x)
def jac(x):
    return np.diag([1, 1, 1]) - getG(x)


# useful to verify your jacobian is correct
def check_jac(my_f, my_J, x, verbose=True):
    """
    Returns True if my_J(x) is close to the true jacobian of my_f(x)
    """
    J = my_J(x)
    J_CS = np.zeros([len(x), len(x)])
    for i in range(len(x)):
        y = np.array(x, np.complex)
        y[i] = y[i] + 1e-10j
        J_CS[:, i] = my_f(y)[:, 0].imag / 1e-10
    if not np.allclose(J, J_CS):
        if verbose:
            print("Jacobian doesn't match - check your function "
                  "and your Jacobian approximation")
            print("The difference between your Jacobian"
                  " and the complex step Jacobian was\n")
            print(J - J_CS)
        return False
    return True


########################################
# Question 3

from scipy.optimize import fsolve


# define any functions you'll use for Q3 here, but call
# them in the main block
def f_2(x, roots=[], advanced=False):
    row_1 = np.sin(x[0]) + np.square(x[1]) + np.log(x[2]) - 3
    row_2 = 3 * x[0] + np.exp2(x[1]) - x[2] ** 3
    row_3 = np.square(x[0]) + np.square(x[1]) + x[2] ** 3 - 6
    to_ret = np.array([row_1, row_2, row_3])
    if roots:
        multiplier = np.ones(3)
        for r in roots:
            if advanced:
                multiplier = np.multiply(multiplier,
                                         np.subtract(1, np.divide(1,
                                                                  np.subtract(x,
                                                                              r))))
            else:
                multiplier = np.multiply(multiplier, np.subtract(x, r))
        if advanced:
            to_ret = np.multiply(to_ret, multiplier)
        else:
            to_ret = np.divide(to_ret, multiplier)
    return to_ret


def naive_search(f, deflation=False, advanced=False):
    roots_found = []
    i = 0
    while len(roots_found) != 4:
        xo = np.random.uniform(-2, 2, 3)
        root = fsolve(f, xo)
        exist = False
        for r in roots_found:
            if np.allclose(r, root):
                exist = True
                break
        if not exist:
            roots_found.append(root)
        if deflation:
            f = lambda x: f_2(x, roots=roots_found, advanced=advanced)
        i += 1
    return roots_found, i


########################################

if __name__ == '__main__':
    # import doctest
    # doctest.testmod()

    # import any other non-standard modules here
    # and run any code for generating your answers here too.

    # Any code calling functions you defined for Q2:

    # here are some sample bits of code you might have if using the
    # provided code:
    if True:
        print("Jac check: ", check_jac(f, jac, np.ones([3, 1])))
        xo = np.ones([3, 1])  # arbitary starting point chosen
        xs = fixed_point(g, xo)
        r, c, errs = conv_info(xs)
        print('Fixed point method has R = {} and C = {}'.format(r, c))
        plt.semilogy(errs, '*-b')  # the "*-" makes it easier to

        xo = np.array([[0], [1], [0]])
        xs = newton(f, jac, xo)
        r, c, errs = conv_info(xs)
        print('Newtons method has R = {} and C = {}'.format(r, c))
        plt.semilogy(errs, '*-r')
        # see convergence behaviour in plot
        xo = np.array([[1], [1], [1]])
        xs = broyden(f, xo)
        r, c, errs = conv_info(xs)
        print('Broydens method has R = {} and C = {}'.format(r, c))
        plt.semilogy(errs, '*-g')
        plt.legend(['Fixed point', 'Newtons', 'Broyden'])
        plt.xlabel('Number of iterations')
        plt.ylabel('Error')
        plt.show()  # make sure to add plot labels if you use this code

    ####################################

    # Any code calling functions you defined for Q3:
    num_samples = []
    times = []
    for _ in range(100):
        start_time = time.perf_counter()
        _, i = naive_search(f_2)
        times.append(time.perf_counter() - start_time)
        num_samples.append(i)
    plt.title("Histogram of number of samples")
    plt.xlabel('Number of samples')
    plt.ylabel('Number of runs')
    plt.hist(num_samples, facecolor='g', alpha=0.75)
    plt.show()

    plt.title("Histogram of time spent")
    plt.xlabel('Time spent')
    plt.ylabel('Number of runs')
    plt.hist(times, facecolor='g', alpha=0.75)
    plt.show()
    print('The number of samples needed '
          'has mean = {} and sigma = {}'.format(np.mean(num_samples),
                                                np.std(num_samples)))
    print('The time taken '
          'has mean = {} and sigma = {}'.format(np.mean(times),
                                                np.std(times)))

    num_samples = []
    times = []
    for _ in range(100):
        start_time = time.perf_counter()
        _, i = naive_search(f_2, deflation=True)
        times.append(time.perf_counter() - start_time)
        num_samples.append(i)
    plt.title("Histogram of number of samples")
    plt.xlabel('Number of samples')
    plt.ylabel('Number of runs')
    plt.hist(num_samples, facecolor='r', alpha=0.75)
    plt.show()

    plt.title("Histogram of time spent")
    plt.xlabel('Time spent')
    plt.ylabel('Number of runs')
    plt.hist(times, facecolor='r', alpha=0.75)
    plt.show()
    print('The number of samples needed '
          'has mean = {} and sigma = {}'.format(np.mean(num_samples),
                                                np.std(num_samples)))
    print('The time taken '
          'has mean = {} and sigma = {}'.format(np.mean(times),
                                                np.std(times)))

    num_samples = []
    times = []
    for _ in range(100):
        start_time = time.perf_counter()
        _, i = naive_search(f_2, deflation=True, advanced=True)
        times.append(time.perf_counter() - start_time)
        num_samples.append(i)
    plt.title("Histogram of number of samples")
    plt.xlabel('Number of samples')
    plt.ylabel('Number of runs')
    plt.hist(num_samples, facecolor='b', alpha=0.75)
    plt.show()

    plt.title("Histogram of time spent")
    plt.xlabel('Time spent')
    plt.ylabel('Number of runs')
    plt.hist(times, facecolor='b', alpha=0.75)
    plt.show()
    print('The number of samples needed '
          'has mean = {} and sigma = {}'.format(np.mean(num_samples),
                                                np.std(num_samples)))
    print('The time taken '
          'has mean = {} and sigma = {}'.format(np.mean(times),
                                                np.std(times)))
