# CSC 336 Summer 2020 HW8 starter code

import numpy as np
import matplotlib.pyplot as plt

########
# Q1 code
########

# the random data
ts = np.linspace(0, 10, 11)
ys = np.random.uniform(-1, 1, ts.shape)

# points to evaluate the spline at
xs = np.linspace(ts[0], ts[-1], 201)


# implement this function
def qintrp_coeffs(ts, ys, c1=0):
    """
    return the coefficents for the quadratic interpolant,
    as specified in the Week 12 worksheet. The coefficients
    should be returned as an array with 3 columns and 1 row
    for each subinterval, so the i'th row contains a_i,b_i,c_i.

    ts are the interpolation points and ys contains the
    data values at each interpolation point

    c1 is the value chosen for c1, default is 0
    """
    # your code here to solve for the coefficients
    a = np.zeros(len(ys) - 1)
    b = np.zeros(len(ys) - 1)
    c = np.zeros(len(ys) - 1)
    c[0] = c1
    for i in range(len(ys) - 1):
        a[i] = ys[i]
        b[i] = np.divide(np.subtract(ys[i + 1], ys[i]),
                         np.subtract(ts[i + 1], ts[i]))
        if i != 0:
            c[i] = np.divide(c[i - 1] * (ts[i] - ts[i - 1]) + b[i - 1] - b[i],
                             ts[i] - ts[i + 1])

    return np.stack([a, b, c]).T


# provided code to evaluate the quadratic spline
def qintrp(coeffs, h, xs):
    """
    Evaluates and returns the quadratic interpolant determined by the
    coeffs array, as returned by qintrp_coeffs, at the points
    in xs.

    h is the uniform space between the knots.

    assumes that each xs is between 0 and h*len(coeffs)
    """
    y = []
    for x in xs:
        i = int(x // h)  # get which subinterval we are in
        if i == len(coeffs):  # properly handle last point
            i = i - 1
        C = coeffs[i]
        ytmp = C[-1] * (x - (i + 1) * h)
        ytmp = (x - i * h) * (ytmp + C[-2])
        ytmp += C[0]
        y.append(ytmp)
    return np.array(y)


# define any additional functions for Q1 here

########################

# define any functions for Q2 and Q3 here
def q2():
    """
    This function returns the coefficients for q2
    (t1, y1) = (-2, -27), (t2, y2) = (0, -1) and (t3, y3) = (1, 0)
    """
    b = np.array([-27, -1, -1, 0, 0, 0, 0, 0])
    A = np.array([[1, -2, 4, -8, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 0, 0, 0, -1, 0, 0],
                  [0, 0, 2, 0, 0, 0, -2, 0],
                  [0, 0, 2, -12, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 2, 6]])
    coeffs = np.linalg.solve(A, b)
    coeffs.resize((2, 4))
    xs = np.linspace(-2, 1, 201)
    y = []
    d1 = []
    d2 = []
    for x in xs:
        if -2 <= x <= 0:
            i = 0
        else:
            i = 1
        a, b, c, d = coeffs[i]
        ytmp = a + b * x + c * x**2 + d * x ** 3
        d1tmp = b + 2 * c * x + 3 * d * x **2
        d2tmp = 2 * c + 6 * d * x
        d1.append(d1tmp)
        d2.append(d2tmp)
        y.append(ytmp)
    y = np.array(y)
    d1 = np.array(d1)
    d2 = np.array(d2)
    ts = np.array([-2, 0, 1])
    ys = np.array([-27, -1, 0])
    plt.title('Cubic Spline Interpolants')
    plt.xlabel('x')
    plt.ylabel('Q(x)')
    plt.plot(xs, y, 'r--')
    plt.plot(xs, d1, 'g--')
    plt.plot(xs, d2, 'y--')
    plt.scatter(ts, ys, c='b')
    plt.legend(['Cubic Spline', 'First derivative',
                'Second derivative', 'Data points'])
    plt.tight_layout()
    plt.show()


def q3():
    """
    This function returns the coefficients for q2
    (t1, y1) = (-2, -27), (t2, y2) = (0, -1) and (t3, y3) = (1, 0)
    """
    b = np.array([-27, -1, -1, 0, 0, 0, 13, 1])
    A = np.array([[1, -2, 4, -8, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1],
                  [0, 1, 0, 0, 0, -1, 0, 0],
                  [0, 0, 2, 0, 0, 0, -2, 0],
                  [0, 1, -4, 12, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 2, 3]])
    coeffs = np.linalg.solve(A, b)
    coeffs.resize((2, 4))
    xs = np.linspace(-2, 1, 201)
    y = []
    d1 = []
    d2 = []
    for x in xs:
        if -2 <= x <= 0:
            i = 0
        else:
            i = 1
        a, b, c, d = coeffs[i]
        ytmp = a + b * x + c * x**2 + d * x ** 3
        d1tmp = b + 2 * c * x + 3 * d * x **2
        d2tmp = 2 * c + 6 * d * x
        d1.append(d1tmp)
        d2.append(d2tmp)
        y.append(ytmp)
    y = np.array(y)
    d1 = np.array(d1)
    d2 = np.array(d2)
    ts = np.array([-2, 0, 1])
    ys = np.array([-27, -1, 0])
    plt.title('Cubic Spline Interpolants')
    plt.xlabel('x')
    plt.ylabel('Q(x)')
    plt.plot(xs, y, 'r--')
    plt.plot(xs, d1, 'g--')
    plt.plot(xs, d2, 'y--')
    plt.scatter(ts, ys, c='b')
    plt.legend(['Cubic Spline', 'First derivative',
                'Second derivative', 'Data points'])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # add any code here calling the functions you defined above
    coeffs = qintrp_coeffs(ts, ys)
    y = qintrp(coeffs, ts[1] - ts[0], xs)
    coeffs = qintrp_coeffs(ts, ys, 0.5)
    y_1 = qintrp(coeffs, ts[1] - ts[0], xs)
    plt.title('Quadratic Spline Interpolants')
    plt.xlabel('x')
    plt.ylabel('Q(x)')
    plt.plot(xs, y, 'r--')
    plt.plot(xs, y_1, 'g--')
    plt.scatter(ts, ys, c='b')
    plt.legend(['Quadratic Spline', 'Quadratic Spline with y0 changed',
                'Data points'])
    plt.tight_layout()
    plt.show()

    q2()
    q3()