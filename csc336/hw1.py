# CSC336 - Homework 1 - starter code

# Q1a


def int2bin(x):
    """
    convert integer x into a binary bit string

    >>> int2bin(0)
    '0'
    >>> int2bin(10)
    '1010'
    """
    to_ret = ''
    while x // 2:
        to_ret = str(x % 2) + to_ret
        x = x // 2
    if x == 1:
        return '1' + to_ret
    return '0' + to_ret


# Q1b
def frac2bin(x):
    """
    convert x into its fractional binary representation.

    precondition: 0 <= x < 1

    >>> frac2bin(0.75)
    '0.11'
    >>> frac2bin(0.625)
    '0.101'
    >>> frac2bin(0.1)
    '0.0001100110011001100110011001100110011001100110011001101'
    >>> frac2bin(0.0)
    '0.0'
    """
    if x == 0.0:
        return '0.0'
    to_ret = '0.'
    while 2 * x != 1.0:
        x *= 2
        if x > 1:
            x -= 1
            to_ret += '1'
        else:
            to_ret += '0'
    return to_ret + '1'


# Question 3
# set these to the values you have chosen as your answers to
# this question. Make sure they aren't just zero and they are
# in the interval (0,1).
x1 = 1 / 2 ** 65
x2 = 1 / 2 ** 200

# Question 4
import numpy as np

# list of floating point data types in numpy for reference
fls = [np.float16, np.float32, np.float64, np.float128]

# fix the following function so that it is correct


# have the correct solution
def eval_with_precision(x, y, fl=np.float64):
    """
    evaluate sin(x**2 + y**2) + 0.1, ensuring that ALL
    calculations are correctly using the
    floating point type fl

    precondition: y != 0

    >>> x = eval_with_precision(2,10,fl=fls[0])
    >>> type(x) == fls[0]
    True
    >>> x == fls[0](-0.6816)
    True
    """
    x, y = fl(x), fl(y)
    a = (x / y) ** fl(2) + fl(2 ** 2)
    return fl(np.sin(a)) + fl(1)/fl(10)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
    # you can add any additional testing you do here
