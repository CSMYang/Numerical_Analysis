import numpy as np
import numpy.linalg as LA
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator, CubicSpline, \
    BarycentricInterpolator
import time

data = np.array([[1900, 76212168], [1910, 92228496],
                 [1920, 106021537], [1930, 123202624],
                 [1940, 132164569], [1950, 151325798],
                 [1960, 179323175], [1970, 203302031],
                 [1980, 226542199]])
data_2 = np.array([[1900, 76212168], [1910, 92228496],
                 [1920, 106021537], [1930, 123202624],
                 [1940, 132164569], [1950, 151325798],
                 [1960, 179323175], [1970, 203302031],
                 [1980, 226542199], [1990, 248709873]])

test_dates = np.linspace(1900, 1980, num=81)


def vand_1(ts):
    A = []
    for t in ts:
        A.append([1, t, t ** 2, t ** 3, t ** 4, t ** 5, t ** 6, t ** 7, t ** 8])
    return np.array(A)


def vand_2(ts):
    ts = np.subtract(ts, 1900)
    A = []
    for t in ts:
        A.append([1, t, t ** 2, t ** 3, t ** 4, t ** 5, t ** 6, t ** 7, t ** 8])
    return np.array(A)


def vand_3(ts):
    ts = np.subtract(ts, 1940)
    A = []
    for t in ts:
        A.append([1, t, t ** 2, t ** 3, t ** 4, t ** 5, t ** 6, t ** 7, t ** 8])
    return np.array(A)


def vand_4(ts):
    ts = np.subtract(ts, 1940)
    ts = np.divide(ts, 40)
    A = []
    for t in ts:
        A.append([1, t, t ** 2, t ** 3, t ** 4, t ** 5, t ** 6, t ** 7, t ** 8])
    return np.array(A)


def newton_basis(ts):
    A = np.zeros((9, 9))
    A[:, 0] = 1
    A[1:, 1] = ts[1] - ts[0], ts[2] - ts[0], ts[3] - ts[0], ts[4] - ts[0], \
               ts[5] - ts[0], ts[6] - ts[0], ts[7] - ts[0], ts[8] - ts[0]
    A[2:, 2] = (ts[2] - ts[0]) * (ts[2] - ts[1]), (ts[3] - ts[0]) * (
            ts[3] - ts[1]), \
               (ts[4] - ts[0]) * (ts[4] - ts[1]), (ts[5] - ts[0]) * (
                       ts[5] - ts[1]), \
               (ts[6] - ts[0]) * (ts[6] - ts[1]), (ts[7] - ts[0]) * (
                       ts[7] - ts[1]), \
               (ts[8] - ts[0]) * (ts[8] - ts[1])
    A[3:, 3] = (ts[3] - ts[0]) * (ts[3] - ts[1]) * (ts[3] - ts[2]), \
               (ts[4] - ts[0]) * (ts[4] - ts[1]) * (ts[4] - ts[2]), \
               (ts[5] - ts[0]) * (ts[5] - ts[1]) * (ts[5] - ts[2]), \
               (ts[6] - ts[0]) * (ts[6] - ts[1]) * (ts[6] - ts[2]), \
               (ts[7] - ts[0]) * (ts[7] - ts[1]) * (ts[7] - ts[2]), \
               (ts[8] - ts[0]) * (ts[8] - ts[1]) * (ts[8] - ts[2])
    A[4:, 4] = (ts[4] - ts[0]) * (ts[4] - ts[1]) * (ts[4] - ts[2]) * (
            ts[4] - ts[3]), \
               (ts[5] - ts[0]) * (ts[5] - ts[1]) * (ts[5] - ts[2]) * (
                       ts[5] - ts[3]), \
               (ts[6] - ts[0]) * (ts[6] - ts[1]) * (ts[6] - ts[2]) * (
                       ts[6] - ts[3]), \
               (ts[7] - ts[0]) * (ts[7] - ts[1]) * (ts[7] - ts[2]) * (
                       ts[7] - ts[3]), \
               (ts[8] - ts[0]) * (ts[8] - ts[1]) * (ts[8] - ts[2]) * (
                       ts[8] - ts[3])
    A[5:, 5] = (ts[5] - ts[0]) * (ts[5] - ts[1]) * (ts[5] - ts[2]) * (
            ts[5] - ts[3]) * (ts[5] - ts[4]), \
               (ts[6] - ts[0]) * (ts[6] - ts[1]) * (ts[6] - ts[2]) * (
                       ts[6] - ts[3]) * (ts[6] - ts[4]), \
               (ts[7] - ts[0]) * (ts[7] - ts[1]) * (ts[7] - ts[2]) * (
                       ts[7] - ts[3]) * (ts[7] - ts[4]), \
               (ts[8] - ts[0]) * (ts[8] - ts[1]) * (ts[8] - ts[2]) * (
                       ts[8] - ts[3]) * (ts[8] - ts[4])
    A[6:, 6] = (ts[6] - ts[0]) * (ts[6] - ts[1]) * (ts[6] - ts[2]) * (
            ts[6] - ts[3]) * (ts[6] - ts[4]) * (ts[6] - ts[5]), \
               (ts[7] - ts[0]) * (ts[7] - ts[1]) * (ts[7] - ts[2]) * (
                       ts[7] - ts[3]) * (ts[7] - ts[4]) * (ts[7] - ts[5]), \
               (ts[8] - ts[0]) * (ts[8] - ts[1]) * (ts[8] - ts[2]) * (
                       ts[8] - ts[3]) * (ts[8] - ts[4]) * (ts[8] - ts[5])
    A[7:, 7] = (ts[7] - ts[0]) * (ts[7] - ts[1]) * (ts[7] - ts[2]) * (
            ts[7] - ts[3]) * (ts[7] - ts[4]) * (ts[7] - ts[5]) * (
                       ts[7] - ts[6]), \
               (ts[8] - ts[0]) * (ts[8] - ts[1]) * (ts[8] - ts[2]) * (
                       ts[8] - ts[3]) * (ts[8] - ts[4]) * (ts[8] - ts[5]) * (
                       ts[8] - ts[6])
    A[8, 8] = (ts[8] - ts[0]) * (ts[8] - ts[1]) * (ts[8] - ts[2]) * (
            ts[8] - ts[3]) * (ts[8] - ts[4]) * (ts[8] - ts[5]) * (
                      ts[8] - ts[6]) * (ts[8] - ts[7])
    return A


def newton_coef(x, y):
    x.astype(float)
    y.astype(float)
    n = len(x)
    a = []
    for i in range(n):
        a.append(y[i])
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            a[i] = np.subtract(a[i], a[i - 1]) / np.subtract(x[i], x[i - j])

    return np.array(a)


def newton_eval(a, x, r):
    x.astype(float)
    n = len(a) - 1
    temp = np.add(a[n], np.subtract(r, x[n]))
    for i in range(n - 1, -1, -1):
        temp = temp * np.subtract(r, x[i]) + a[i]
    return temp


def f(t, coeffs):
    coeffs = coeffs[::-1]
    t = np.subtract(t, 1940)
    t = np.divide(t, 40)
    P = coeffs[0]
    for i in range(1, len(coeffs)):
        P = np.multiply(P, t) + coeffs[i]
    return P


def f_newton(t, ts, coeffs):
    to_ret = np.array([1, t - ts[0], (t - ts[0]) * (t - ts[1]),
                       (t - ts[0]) * (t - ts[1]) * (t - ts[2]),
                       (t - ts[0]) * (t - ts[1]) * (t - ts[2]) * (t - ts[3]),
                       (t - ts[0]) * (t - ts[1]) * (t - ts[2]) * (t - ts[3]) * (
                               t - ts[4]),
                       (t - ts[0]) * (t - ts[1]) * (t - ts[2]) * (t - ts[3]) *
                       (t - ts[4]) * (t - ts[5]),
                       (t - ts[0]) * (t - ts[1]) * (t - ts[2]) * (t - ts[3]) *
                       (t - ts[4]) * (t - ts[5]) * (t - ts[6]),
                       (t - ts[0]) * (t - ts[1]) * (t - ts[2]) * (t - ts[3]) *
                       (t - ts[4]) * (t - ts[5]) * (t - ts[6]) * (t - ts[7])])
    return np.sum(np.multiply(to_ret, coeffs))


class Newton:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def run(self, tmp_x):
        table = np.zeros([len(self.x), len(self.x) + 1], dtype=float)

        for i in range(len(self.x)):
            table[i][0] = self.x[i]
            table[i][1] = self.y[i]
        for i in range(2, table.shape[1]):
            for j in range(i - 1, table.shape[0]):
                table[j][i] = (table[j][i - 1] - table[j - 1][i - 1]) / \
                              (self.x[j] - self.x[j - i + 1])
        tmp_y = []
        for ans_x in tmp_x:
            ans_y = 0
            for i in range(table.shape[0]):
                tmp = table[i][i + 1]
                for j in range(i):
                    tmp *= (ans_x - self.x[j])
                ans_y += tmp
            tmp_y.append(ans_y)
        return tmp_y


if __name__ == '__main__':
    # Q1
    A_1 = vand_1(data[:, 0])
    A_2 = vand_2(data[:, 0])
    A_3 = vand_3(data[:, 0])
    A_4 = vand_4(data[:, 0])
    cond_1 = LA.cond(A_1, 1)
    cond_2 = LA.cond(A_2, 1)
    cond_3 = LA.cond(A_3, 1)
    cond_4 = LA.cond(A_4, 1)
    print('Condition number for basis 1 is: {}'.format(cond_1))
    print('Condition number for basis 2 is: {}'.format(cond_2))
    print('Condition number for basis 3 is: {}'.format(cond_3))
    print('Condition number for basis 4 is: {}'.format(cond_4))
    # Q2
    coeffs = LA.solve(A_4, data[:, 1])
    ys = []
    for t in test_dates:
        y = f(t, coeffs)
        ys.append(y)
    ys = np.array(ys)
    plt.title('Interpolants')
    plt.xlabel('year')
    plt.ylabel('population')
    plt.plot(test_dates, ys, 'r--')
    # Q3
    Pchip = PchipInterpolator(data[:, 0], data[:, 1])
    ys = Pchip(test_dates)
    plt.plot(test_dates, ys, 'g')
    # Q4
    cs = CubicSpline(data[:, 0], data[:, 1])
    ys = cs(test_dates)
    plt.plot(test_dates, ys, 'b--')
    plt.scatter(data[:, 0], data[:, 1], c='y')
    plt.legend(['Polynomial', 'Hermite Cubic', 'Cubic spline', 'Data points'])
    plt.show()
    # Q5
    true_pop = 248709873
    poly_extra = f(1990, coeffs)
    poly_rele = abs(np.divide(np.subtract(poly_extra, true_pop), true_pop))
    hermite_extra = Pchip(1990)
    hermite_rele = abs(np.divide(np.subtract(hermite_extra, true_pop),
                                 true_pop))
    cubic_extra = cs(1990)
    cubic_rele = abs(np.divide(np.subtract(cubic_extra, true_pop), true_pop))
    print('Extrapolations:')
    print('Polynomial: {}, relative error: {}'.format(poly_extra, poly_rele))
    print('Hermite Cubic: {}, relative error: {}'.format(hermite_extra,
                                                         hermite_rele))
    print('Cubic Spline: {}, relative error: {}'.format(cubic_extra,
                                                        cubic_rele))
    # Q6
    start_time = time.perf_counter()
    _ = f(test_dates, coeffs)
    time_spent = time.perf_counter() - start_time
    print("Time spent for Polynomial Interpolant is: {}".format(time_spent))
    lagrange = BarycentricInterpolator(data[:, 0], data[:, 1])
    start_time = time.perf_counter()
    _ = lagrange(test_dates)
    time_spent = time.perf_counter() - start_time
    print("Time spent for Lagrange Interpolant is: {}".format(time_spent))
    start_time = time.perf_counter()
    _ = cs(test_dates)
    time_spent = time.perf_counter() - start_time
    print("Time spent for Cubic Spline Interpolant is: {}".format(time_spent))
    # Q7
    test_date = np.linspace(1900, 1990, num=91)
    newton = Newton(data_2[:, 0], data_2[:, 1])
    ys = newton.run(test_date)
    plt.plot(test_date, ys, 'g--')

    test_date = np.linspace(1900, 1990, num=91)
    newton = Newton(data[:, 0], data[:, 1])
    ys = newton.run(test_date)
    plt.title('Newton Basis Polynomial Interpolants')
    plt.xlabel('year')
    plt.ylabel('Population')
    plt.plot(test_date, ys, 'r--')
    plt.scatter(data[:, 0], data[:, 1], c='y')
    plt.legend(['Degree 9', 'Degree 8', 'Data points'])
    plt.show()
    # Q8
    rounded_data = np.around(data[:, 1], -6)
    new_coeffs = LA.solve(A_4, rounded_data)
    print('Coefficients got from rounded data:')
    print(new_coeffs)
    print('Coefficients got from unrounded data:')
    print(coeffs)