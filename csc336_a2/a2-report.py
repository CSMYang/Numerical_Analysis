# CSC336 Assignment #2 starter code for the report question

# These are some basic imports you will probably need,
# but feel free to add others here if you need them.
import numpy as np
from scipy.sparse import diags
import scipy.linalg as sla
from scipy.sparse.linalg import spsolve, spsolve_triangular
from scipy.linalg import solve, solve_triangular, solve_banded
import time
import matplotlib
import matplotlib.pyplot as plt

"""
See the examples in class this week or ask on Piazza if you
aren't sure how to start writing the code
for the report questions.

We won't be grading your code unless there is something
strange with your output, so don't worry too much about
coding style.

Please start early and ask questions.
"""

"""
timing code sample (feel free to use timeit if you find it easier)
#you might want to run this in a loop and record
#the average or median time taken to get a more reliable timing
start_time = time.perf_counter()
#your code here to time
time_taken = time.perf_counter() - start_time
"""


def get_true_sol(n):
    """
    returns the true solution of the continuous model on the mesh,
    x_i = i / n , i=1,2,...,n.
    """
    x = np.linspace(1 / n, 1, n)
    d = (1 / 24) * (-(1 - x) ** 4 + 4 * (1 - x) - 3)
    return d


def compare_to_true(d):
    """
    produces plot similar to the handout,
    the input is the solution to the n x n banded linear system,
    this is one way to visually check if your code is correct.
    """
    dtrue = get_true_sol(100)  # use large enough n to make plot look smooth

    plt.tight_layout()
    matplotlib.rcParams.update({'font.size': 14})
    plt.title("Horizontal Cantilevered Bar")
    plt.xlabel("x")
    plt.ylabel("d(x)")

    xtrue = np.linspace(1 / 100, 1, 100)
    plt.plot(xtrue, dtrue, 'k')

    n = len(d)
    x = np.linspace(0, 1, n + 1)
    plt.plot(x, np.hstack([0, d]), '--r')

    plt.legend(['exact', str(n)])
    plt.grid()
    plt.show()


def time_cost_experiment(ns):
    """
    This function does the time cost experiment
    """
    times = {'lu': [], 'banded': [], 'sparse_lu': [], 'r': [], 'banded_r': [],
             'sparse_r': [], 'chol': []}

    for n in ns:

        diags_2 = np.ones(n - 2)
        diags_1 = np.ones(n - 1) * -4
        diags_1[-1] = -2
        diags_0 = np.ones(n) * 6
        diags_0[0] = 9
        diags_0[-1] = 1
        diags_0[-2] = 5
        m = diags([diags_2, diags_1, diags_0, diags_1, diags_2],
                  [-2, -1, 0, 1, 2], format='csr')
        m_dense = m.toarray()
        b = np.divide(np.ones(n), -(n ** 4))

        # LU
        start_time = time.perf_counter()
        x = sla.solve(m_dense, b)
        time_taken = time.perf_counter() - start_time
        times['lu'].append(time_taken)

        # Banded LU
        ab = np.zeros((5, n))
        ab[0, 2:] = m_dense.diagonal(offset=2)[:]
        ab[1, 1:] = m_dense.diagonal(offset=1)[:]
        ab[2, :] = m_dense.diagonal()[:]
        ab[3, :-1] = m_dense.diagonal(offset=-1)[:]
        ab[4, :-2] = m_dense.diagonal(offset=-2)[:]
        start_time = time.perf_counter()
        x = sla.solve_banded((2, 2), ab, b)
        time_taken = time.perf_counter() - start_time
        times['banded'].append(time_taken)

        # Sparse LU
        start_time = time.perf_counter()
        x = spsolve(m, b)
        time_taken = time.perf_counter() - start_time
        times['sparse_lu'].append(time_taken)

        # Prefactored
        diag_2 = np.ones(n - 2)
        diag_1 = np.ones(n - 1) * -2
        diag_0 = np.ones(n)
        diag_0[0] = 2
        r = diags([diag_0, diag_1, diag_2], [0, 1, 2], format='csr')
        r_dense = r.toarray()
        start_time = time.perf_counter()
        y = sla.solve_triangular(r_dense, b)
        x = sla.solve_triangular(r_dense.T, y, lower=True)
        time_taken = time.perf_counter() - start_time
        times['r'].append(time_taken)

        # Banded prefactored
        ab_1 = np.zeros((3, n))
        ab_1[0, 2:] = np.ones(n - 2)[:]
        ab_1[1, 1:] = np.ones(n - 1)[:] * -2
        ab_1[2, :] = np.ones(n)[:]
        ab_1[2, 0] = 2
        ab_2 = np.zeros((3, n))
        ab_2[0, :] = ab_1[2, :]
        ab_2[1, : -1] = ab_1[1, 1:]
        ab_2[2, :-2] = ab_1[0, 2:]
        start_time = time.perf_counter()
        y = sla.solve_banded((0, 2), ab_1, b)
        x = sla.solve_banded((2, 0), ab_2, y)
        time_taken = time.perf_counter() - start_time
        times['banded_r'].append(time_taken)

        # Sparse prefactored
        start_time = time.perf_counter()
        y = spsolve_triangular(r, b, lower=False)
        x = spsolve_triangular(r.T, y, lower=True)
        time_taken = time.perf_counter() - start_time
        times['sparse_r'].append(time_taken)

        # Cholesky
        start_time = time.perf_counter()
        x = sla.cho_solve(sla.cho_factor(m_dense), b)
        time_taken = time.perf_counter() - start_time
        times['chol'].append(time_taken)

    legend = []
    plt.figure()
    plt.title('Time cost growth')
    plt.xlabel('width of the square matrix')
    plt.ylabel('time taken')
    for key in times:
        legend.append(key)
        plt.plot(ns, times[key])
    plt.legend(legend)
    plt.grid(axis='y')
    plt.show()


def relative_error_experiment(ns):
    """
    This function does the relative error experiment
    """
    rele_errs = {'banded': [], 'sparse_lu': [], 'r': [],
                 'banded_r': [], 'sparse_r': [], 'chol': []}
    for n in ns:
        # True solution
        t_sol = get_true_sol(n)

        diags_2 = np.ones(n - 2)
        diags_1 = np.ones(n - 1) * -4
        diags_1[-1] = -2
        diags_0 = np.ones(n) * 6
        diags_0[0] = 9
        diags_0[-1] = 1
        diags_0[-2] = 5
        m = diags([diags_2, diags_1, diags_0, diags_1, diags_2],
                  [-2, -1, 0, 1, 2], format='csr')
        m_dense = m.toarray()
        b = np.divide(np.ones(n), -(n ** 4))

        # Banded LU
        ab = np.zeros((5, n))
        ab[0, 2:] = m_dense.diagonal(offset=2)[:]
        ab[1, 1:] = m_dense.diagonal(offset=1)[:]
        ab[2, :] = m_dense.diagonal()[:]
        ab[3, :-1] = m_dense.diagonal(offset=-1)[:]
        ab[4, :-2] = m_dense.diagonal(offset=-2)[:]
        x = sla.solve_banded((2, 2), ab, b)
        rele_errs['banded'].append(abs(np.divide(sla.norm(abs(np.subtract(x, t_sol))),
                                                 sla.norm(t_sol))))

        # Sparse LU
        x = spsolve(m, b)
        rele_errs['sparse_lu'].append(
            abs(np.divide(sla.norm(abs(np.subtract(x, t_sol))),
                          sla.norm(t_sol))))

        # Prefactored
        diag_2 = np.ones(n - 2)
        diag_1 = np.ones(n - 1) * -2
        diag_0 = np.ones(n)
        diag_0[0] = 2
        r = diags([diag_0, diag_1, diag_2], [0, 1, 2], format='csr')
        r_dense = r.toarray()
        y = sla.solve_triangular(r_dense, b)
        x = sla.solve_triangular(r_dense.T, y, lower=True)
        rele_errs['r'].append(
            abs(np.divide(sla.norm(abs(np.subtract(x, t_sol))),
                          sla.norm(t_sol))))

        # Banded prefactored
        ab_1 = np.zeros((3, n))
        ab_1[0, 2:] = np.ones(n - 2)[:]
        ab_1[1, 1:] = np.ones(n - 1)[:] * -2
        ab_1[2, :] = np.ones(n)[:]
        ab_1[2, 0] = 2
        ab_2 = np.zeros((3, n))
        ab_2[0, :] = ab_1[2, :]
        ab_2[1, : -1] = ab_1[1, 1:]
        ab_2[2, :-2] = ab_1[0, 2:]
        y = sla.solve_banded((0, 2), ab_1, b)
        x = sla.solve_banded((2, 0), ab_2, y)
        rele_errs['banded_r'].append(
            abs(np.divide(sla.norm(abs(np.subtract(x, t_sol))),
                          sla.norm(t_sol))))

        # Sparse prefactored
        y = spsolve_triangular(r, b, lower=False)
        x = spsolve_triangular(r.T, y, lower=True)
        rele_errs['sparse_r'].append(
            abs(np.divide(sla.norm(abs(np.subtract(x, t_sol))),
                          sla.norm(t_sol))))

        # Cholesky
        x = sla.cho_solve(sla.cho_factor(m_dense), b)
        rele_errs['chol'].append(
            abs(np.divide(sla.norm(abs(np.subtract(x, t_sol))),
                          sla.norm(t_sol))))

    legend = []
    plt.figure()
    plt.title('Relative Errors of each method')
    plt.xlabel('Width of the square matrix')
    plt.ylabel('Relative error')
    for key in rele_errs:
        legend.append(key)
        plt.loglog(ns, rele_errs[key])
    plt.legend(legend)
    plt.grid(axis='y')
    plt.show()


def matrix_computation_experiment():
    """
    This function does the matrix computation experiment
    """
    ns = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
    times = {'Direct computation':[], 'Improved computation':[]}
    for n in ns:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        C = np.random.rand(n, n)
        I = np.eye(n)
        b = np.random.rand(n)

        start_time = time.perf_counter()
        inv_B = np.linalg.inv(B)
        inv_C = np.linalg.inv(C)
        _ = inv_B @ (2 * A + I) @ (inv_C + A) @ b
        time_taken = time.perf_counter() - start_time
        times['Direct computation'].append(time_taken)

        start_time = time.perf_counter()
        k = sla.solve(C, b)
        m = A + A + I
        n = (k + (A @ b))
        b_1 = m @ n
        _ = sla.solve(B, b_1)
        time_taken = time.perf_counter() - start_time
        times['Improved computation'].append(time_taken)

    n = 10000
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    C = np.random.rand(n, n)
    I = np.eye(n)
    b = np.random.rand(n)

    start_time = time.perf_counter()
    inv_B = np.linalg.inv(B)
    inv_C = np.linalg.inv(C)
    _ = inv_B @ (2 * A + I) @ (inv_C + A) @ b
    time_taken = time.perf_counter() - start_time
    print('Time taken for direct computation '
          'when n = 10000: {}'.format(time_taken))

    start_time = time.perf_counter()
    k = sla.solve(C, b)
    m = A + A + I
    n = (k + (A @ b))
    b_1 = m @ n
    _ = sla.solve(B, b_1)
    time_taken = time.perf_counter() - start_time
    print('Time taken for the improved method '
          'when n = 10000: {}'.format(time_taken))

    legend = []
    plt.figure()
    plt.title('Matrix computation experiment')
    plt.xlabel('Width of the square matrix')
    plt.ylabel('Time Taken')
    for key in times:
        legend.append(key)
        plt.plot(ns, times[key])
    plt.legend(legend)
    plt.grid(axis='y')
    plt.show()


if __name__ == "__main__":
    # # Q1
    # ns = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
    # time_cost_experiment(ns)
    # # Q2
    # ns = [16, 32, 64, 128, 256, 512, 1024, 2048,
    #       4096, 8192, 16384]
    # relative_error_experiment(ns)
    matrix_computation_experiment()

