# CSC 336 HW#3 starter code

import scipy.linalg as sla
import numpy as np
import numpy.linalg as LA

# Q1 assign values to the following variables as
# specified in the handout

A = np.array(
    [[21.0, 67.0, 88.0, 73.0], [76.0, 63.0, 7.0, 20.0], [0.0, 85.0, 56.0, 54.0],
     [19.3, 43.0, 30.2, 29.4]])
B = np.array([[1, 2, 3, 4], [12, 13, 14, 5], [11, 16, 15, 6], [10, 9, 8, 7]])
C = B.T
b = np.array([141.0, 109.0, 218.0, 93.7])
x = sla.solve(A, b)
r = np.subtract(np.dot(A, x), b)
Ainv = sla.inv(A)
c1 = LA.cond(A, 1)
c1_2 = float(np.multiply(LA.norm(A, 1), LA.norm(Ainv, 1)))
cinf = LA.cond(A, np.inf)
A32 = A.astype(np.float32)
b32 = b.astype(np.float32)
x32 = sla.solve(A32, b32)
y = np.dot((LA.inv(B) @ (2 * A + np.eye(A.shape[0])) @ (LA.inv(C) + A)), b)


# Q2 Hilbert matrix question
# your code here
def hilbert_experiment():
    """
    This function tries different sizes of the hilbert matrices and prints
    out the relative error and condition number of the hilbert matrices
    until the relative error is greater than 100%.
    """
    print('Results for float64')
    print('n |   rel err | cond(H)')
    print('-----------------------')
    n = 2
    hilbert = sla.hilbert(n).astype(np.float64)
    cond = LA.cond(hilbert, np.inf)
    answer = np.ones(n).astype(np.float64)
    b = np.dot(hilbert, answer).astype(np.float64)
    computed_answer = sla.solve(hilbert, b)
    answer_norm = LA.norm(answer, np.inf)
    computed_norm = LA.norm(computed_answer, np.inf)
    rel_err = abs(np.divide(answer_norm - computed_norm, answer_norm))
    print('{} | {:.3e} | {:.3e}'.format(n, rel_err, cond))
    while rel_err < 1:
        n += 1
        hilbert = sla.hilbert(n).astype(np.float64)
        cond = LA.cond(hilbert, np.inf)
        answer = np.ones(n).astype(np.float64)
        b = np.dot(hilbert, answer).astype(np.float64)
        computed_answer = sla.solve(hilbert, b)
        answer_norm = LA.norm(answer, np.inf)
        computed_norm = LA.norm(computed_answer, np.inf)
        rel_err = abs(np.divide(answer_norm - computed_norm, answer_norm))
        print('{} | {:.3e} | {:.3e}'.format(n, rel_err, cond))


# Q3c provided code for gaussian elimination (implements algorithm from the
# GE notes)
def ge(A, b):
    for k in range(A.shape[0] - 1):
        for i in range(k + 1, A.shape[1]):
            if A[k, k] != 0:
                A[i, k] = A[i, k] / A[k, k]
            else:
                return False
            A[i, k + 1:] -= A[i, k] * A[k, k + 1:]
            b[i] = b[i] - A[i, k] * b[k]
    return True


def bs(A, b):
    x = np.zeros(b.shape)
    x[:] = b[:]
    for i in range(A.shape[0] - 1, -1, -1):
        for j in range(i + 1, A.shape[0]):
            x[i] -= A[i, j] * x[j]
        if A[i, i] != 0:
            x[i] /= A[i, i]
        else:
            return None
    return x


def ge_solve(A, b):
    if ge(A, b):
        return bs(A, b)
    else:
        return None  # GE failed


def solve(eps):
    """
    return the solution of [ eps 1 ] [x1]   [1 + eps]
                           |       | |  | = |       |
                           [ 1   1 ] [x2]   [   2   ]
    The solution is obtained using GE without pivoting
    and back substitution. (the provided ge_solve above)
    """
    A = np.array([[eps, 1], [1, 1]])
    b = np.array([1 + eps, 2])
    x = ge_solve(A, b)
    return x


# Q3d code here for generating your table of values
def ge_experiment():
    """
    This function does the experiment for Gaussian elimination algorithm.
    """
    print('Results of GE experiment:')
    print('k |   rel err | epsilon')
    for k in range(1, 11):
        eps = 10 ** (-2 * k)
        results = solve(eps)
        result_norm = LA.norm(results)
        answer_norm = LA.norm([1, 1])
        rel_err = abs(np.divide(answer_norm - result_norm, answer_norm))
        print('{} | {:.3e} | {:.3e}'.format(k, rel_err, eps))


if __name__ == '__main__':
    hilbert_experiment()
    print()
    ge_experiment()
