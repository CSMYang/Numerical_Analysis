# CSC336 Assignment #1 starter code
import numpy as np


# Q2a
def alt_harmonic(fl=np.float16):
    """
    Returns [n, s], where n is the last term that was added
    to the sum, s, of the alternating harmonic series.

    The floating point type fl is used in the calculations.
    """
    sum = fl(0)
    n = 1
    current_term = np.divide((-1) ** (n + 1), n)
    while sum != fl(np.add(sum, current_term)):
        sum = fl(np.add(sum, current_term))
        n += 1
        current_term = np.divide((-1) ** (n + 1), n)
    return [n - 1, sum]


# Q2b
# add code here as stated in assignment handout
q2b_rel_error = abs(np.subtract(1, np.divide(alt_harmonic()[1], np.log(2))))


# Q2c
def alt_harmonic_given_m(m, fl=np.float16):
    """
    Returns the sum of the first m terms of the alternating
    harmonic series. The sum is performed in an appropriate
    order so as to reduce rounding error.

    The floating point type fl is used in the calculations.
    """
    pos_sum = 0
    nega_sum = 0
    for i in range(1, m):
        if i % 2:
            pos_sum += np.divide(1, i)
        else:
            nega_sum += np.divide(1, i)
    return fl(np.subtract(pos_sum, nega_sum))


# Q3a
def alt_harmonic_v2(fl=np.float32):
    """
    Returns [n, s], where n is the last term that was added
    to the sum, s, of the alternating harmonic series (using
    the formula in Q3a, where terms are paired).

    The floating point type fl is used in the calculations.
    """
    sum = fl(0)
    n = 1
    current_term = np.divide(1, np.multiply(2 * n, np.subtract(2 * n, 1)))
    while sum != fl(np.add(sum, current_term)):
        sum = fl(np.add(sum, current_term))
        n += 1
        current_term = np.divide(1, np.multiply(2 * n, np.subtract(2 * n, 1)))
    return [n - 1, sum]


# Q3b
# add code here as stated in assignment handout
q3b_rel_error = abs(np.subtract(1, np.divide(alt_harmonic_v2()[1], np.log(2))))


# Q4b
def hypot(a, b):
    """
    Returns the hypotenuse, given sides a and b.
    """
    long = max(a, b)
    short = min(a, b)
    r = np.divide(short, long)
    return np.multiply(long, np.sqrt(np.add(1, r ** 2)))


# Q4c
q4c_input = [1/2.0 ** 200]
# see handout for what value should go here.

if __name__ == "__main__":
    # import doctest
    # doctest.testmod()
    n, s = alt_harmonic()
    print(n, s)
    print(np.log(2))
    print(q2b_rel_error)
    a = alt_harmonic_given_m(4096)
    print(a)
    print(alt_harmonic_v2())
