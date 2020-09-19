# CSC336 Homework #2 starter code
import numpy as np


def arccosh(y):
    """ return an accurate approximation of arccosh(y),
        using the algorithm you came up with on this week's
        worksheet.
        y is a python double precision floating point number
        (you may not use any builtin arccosh function)
    >>> np.allclose(arccosh(1e20),np.arccosh(1e20),\
                    rtol = 1e-14, atol=0)
    True
    >>> np.allclose( arccosh(1+1e-5),np.arccosh(1+1e-5),\
                    rtol = 1e-14, atol=0)
    True
    >>> np.allclose(arccosh(np.cosh(np.pi)),np.arccosh(np.cosh(np.pi)),\
                    rtol = 1e-14, atol=0)
    True
    >>> np.allclose(arccosh(1e200),np.arccosh(1e200),\
                    rtol = 1e-14, atol=0)
    True
    """
    if y >= 10 or y <= 0.1:
        return np.log(np.add(y, np.multiply(np.sqrt(np.add(y, 1)),
                                            np.sqrt(np.subtract(y, 1)))))
    z = np.subtract(y, 1)
    return np.log1p(np.add(z, np.multiply(np.sqrt(np.add(y, 1)),
                                            np.sqrt(z))))


# Q2 floating point calculations
#
# Complete each of the following calculations using a base
# 10 floating point number system with 3 digit precision
# and one digit exponent (i.e. in the range $[-9,9]$).
# Use the notation '0.123e2' when entering the number
# 12.3 = 0.123 \times 10^2.
# Enter 'overflow' if overflow would occur
# Run the autotests on MarkUs to check your answers
# Let a = 0.797e0 and b = 0.799e0$.
x1 = 'overflow'  # 0.600e9 + 0.500e9

x2 = '0.122e0'  # 0.122e0 + 0.199e-3

x3 = '0.800e0'  # (a+b) / 0.200e1

x4 = '0.400e0'  # a + (b-a) / 0.200e1


def harmonic(fl=np.float16):
    """
    sums the harmonic series until the sum stops changing
    and returns a list containing the last value of n added
    to the sum and the final sum

    Note: The student autotest on MarkUs only runs harmonic(),
    since harmonic(np.float32) takes a bit longer to run,
    but the graded tests will both be the same as the doctests
    below.

    If you are getting different output, please ask on piazza.

    >>> n,s = harmonic()
    >>> abs(n-513) <= 1 and np.allclose(s,7.086)
    True
    >>> n,s = harmonic(np.float32)
    >>> abs(n-2097152) <= 1 and np.allclose(s,15.403683)
    True
    """
    s = 0
    n = 1
    while s < fl(s + 1 / n):
        s = fl(s + 1 / n)
        n += 1
    return n, s


if __name__ == "__main__":
    import doctest

    doctest.testmod()
