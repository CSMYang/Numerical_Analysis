import numpy as np
from scipy.linalg import inv
import numpy.linalg as LA
import time
import matplotlib.pyplot as plt
from scipy.stats import ortho_group


def newton_inverse(A, atol=1e-14):
    """
    This function uses the newton's iterative scheme to find the inverse of
    matrix A
    """
    xo = np.divide(A.T, np.multiply(LA.norm(A, 1), LA.norm(A, np.inf)))
    I = np.eye(A.shape[0])
    current_x = xo
    residual = LA.norm(np.subtract(I, np.matmul(A, current_x)))
    while residual > atol:
        current_x = current_x + np.matmul(current_x,
                                          np.subtract(I,
                                                      np.matmul(A, current_x)))
        residual = LA.norm(np.subtract(I, np.matmul(A, current_x)))

    return current_x, residual


if __name__ == '__main__':
    newton_residuals = []
    inv_residuals = []
    newton_times = []
    inv_times = []
    ns = np.geomspace(2, 512, num=9).astype(int)
    for i in range(9):
        n = ns[i]
        A = ortho_group.rvs(n)
        I = np.eye(n)

        start_time = time.perf_counter()
        _, residual = newton_inverse(A)
        time_taken = time.perf_counter() - start_time
        newton_residuals.append(residual)
        newton_times.append(time_taken)

        start_time = time.perf_counter()
        inv_A = inv(A)
        time_taken = time.perf_counter() - start_time
        residual = LA.norm(np.subtract(I, np.matmul(A, inv_A)))
        inv_residuals.append(residual)
        inv_times.append(time_taken)

    plt.title('Time taken vs Size of matrix')
    plt.xlabel('Size of matrix')
    plt.ylabel('Time taken')
    plt.plot(ns, newton_times)
    plt.plot(ns, inv_times)
    plt.legend(['Newtons method', 'scipy.linalg.inv'])
    plt.show()

    plt.title('Norm of residual vs Size of matrix')
    plt.xlabel('Size of matrix')
    plt.ylabel('Norm of residual')
    plt.plot(ns, newton_residuals)
    plt.plot(ns, inv_residuals)
    plt.legend(['Newtons method', 'scipy.linalg.inv'])
    plt.show()