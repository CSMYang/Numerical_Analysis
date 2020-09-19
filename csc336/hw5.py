#CSC336 Summer 2020 HW5 starter code

import numpy as np
import matplotlib.pyplot as plt

#question 1
def estimate_convergence_rate(xs):
    """
    Return approximate values of the convergence rate r and
    constant C (see section 5.4 in Heath for definition and
    this week's worksheet for the approach), for the given
    sequence, xs, of x values.

    (You might find np.diff convenient)

    Examples:

    >>> xs = 1+np.array([1,0.5,0.25,0.125,0.0625,0.03125])
    >>> b,c = estimate_convergence_rate(xs)
    >>> close = lambda x,y : np.allclose(x,y,atol=0.1)
    >>> close(b,1)
    True
    >>> close(c,0.5)
    True
    >>> xs = [0, 1.0, 0.7357588823428847, 0.6940422999189153,\
    0.6931475810597714, 0.6931471805600254, 0.6931471805599453]
    >>> b,c = estimate_convergence_rate(xs)
    >>> close(b,2)
    True
    >>> close(c,0.5)
    True
    """
    es = np.diff(xs)
    e_0, e_1, e_2 = abs(es[-3:])
    a = np.divide(e_1, e_0)
    b = np.divide(e_2, e_1)
    r = np.divide(np.log(b), np.log(a))
    c = np.divide(e_2, e_1 ** r)
    return r, c

#question 2
#Put any functions you define for Q2 here.
#See the worksheet for a similar example if you have
#trouble getting started with Q2.

def g_1():
    """
    This function plots the change of estimate value
    """
    x = np.float64(1.5)
    xs_1 = [x]
    for _ in range(100):
        x = np.divide(2 + np.square(x), 3)
        xs_1.append(x)

    x = np.float64(1.5)
    xs_2 = [x]

    for _ in range(90):
        x = np.sqrt(np.subtract(np.multiply(x, 3), 2))
        xs_2.append(x)

    x = np.float64(1.5)
    xs_3 = [x]
    for _ in range(50):
        x = np.subtract(3, np.divide(2, x))
        xs_3.append(x)

    x = np.float64(3)
    xs_4 = [x]
    for _ in range(6):
        x = np.divide(np.square(x) - 2, 2 * x - 3)
        xs_4.append(x)

    plt.title('Value of x at each iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Value of x')
    plt.plot(np.arange(0, 101, 1), xs_1)
    plt.plot(np.arange(0, 91, 1), xs_2)
    plt.plot(np.arange(0, 51, 1), xs_3)
    plt.plot(np.arange(0, 7, 1), xs_4)
    plt.legend(['g_1', 'g_2', 'g_3', 'g_4'])
    plt.show()
    r, c = estimate_convergence_rate(xs_2)
    print('R is {} and C is {}'.format(r, c))
    r, c = estimate_convergence_rate(xs_3)
    print('R is {} and C is {}'.format(r, c))
    r, c = estimate_convergence_rate(xs_4)
    print('R is {} and C is {}'.format(r, c))

#optional question 3 starter code
a = np.pi #value of root
def f(x,m=2):
    return (x - a)**m
def df(x,m=2):
    return m*(x-a)**(m-1)

def newton(f,fp,xo,atol = 1e-10):
    """
    Simple implementation of Newton's method for scalar problems,
    with stopping criterion based on absolute difference between
    values of x on consecutive iterations.

    Returns the sequence of x values.
    """
    x = xo
    xs = [xo]
    err = 1
    while(err > atol):
        x = x - f(x)/fp(x)
        xs.append(x)
        err = np.abs(xs[-2]-xs[-1])
    return xs

def multiplicity_newton(f,fp,xo,atol = 1e-10):
    """
    version of Newton's method that monitors the convergence rate,
    r and/or C, and tries to speedup convergence for multiple roots

    Returns the sequence of x values.
    """
    #modify the Newton's method code below based on your algorithm
    #from Q3b
    x = xo
    xs = [xo]
    err = 1
    while(err > atol):
        x = x - f(x)/fp(x)
        xs.append(x)
        err = np.abs(xs[-2]-xs[-1])
    return xs


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    #call any code related to Q2 here
    g_1()
    # xs = g_2()
    # r, c = estimate_convergence_rate(xs)
    # print('R is {} and C is {}'.format(r, c))
    # xs = g_3()
    # r, c = estimate_convergence_rate(xs)
    # print('R is {} and C is {}'.format(r, c))


    #optional code if you try Q3c
    if False:
        xo = 0
        print("error orig, error alt, iters orig, iters alt")
        for m in range(2,8):
            xs = multiplicity_newton(lambda x: f(x,m=m),lambda x: df(x,m=m),xo)
            iters_alt = len(xs)
            xalt = xs[-1]
            xs = newton(lambda x: f(x,m=m),lambda x: df(x,m=m),xo)
            iters_orig = len(xs)
            x = xs[-1]
            print(f"{x-a:<10.2e},{xalt-a:<10.2e},{iters_orig:<11d},"
                    f"{iters_alt:<10d}")