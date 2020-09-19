# CSC 336 HW#4 starter code

import scipy.linalg as sla
import numpy as np

# Q1 - set these to their correct values
M_35 = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                 [0, 0.5, 1, 0], [0, -2, 0, 1]])

P_36_a = np.array([[1, 0, 0, 0], [0, 0, 0, 1],
                   [0, 0, 1, 0], [0, 1, 0, 0]])
P_36_b = np.array([[0, 0, 0, 1], [0, 0, 1, 0],
                   [0, 1, 0, 0], [1, 0, 0, 0]])

A_q1c = np.array([[2, 3, -6], [0.5, -7.5, 11], [1.5, 13/15, 7/15]])
y_q1c = np.array([-8, 11, 7/15])
x_q1c = np.array([-1, 0, 1])


# Q3
def q3(A, B, C, b):
    k = sla.lu_solve(sla.lu_factor(C), b)
    y = 2 * np.dot(A, k) + 2 * np.dot(np.matmul(A, A), b) + k + np.dot(A, b)
    x = sla.lu_solve(sla.lu_factor(B), y)
    return x


if __name__ == '__main__':
    pass
