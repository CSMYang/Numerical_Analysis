# CSC336 Assignment #2 starter code

import numpy as np
import scipy.linalg as sla


# Q4a
def p_to_q(p):
    """
    return the permutation vector, q, corresponding to
    the pivot vector, p.
    >>> p_to_q(np.array([2,3,2,3]))
    array([2, 3, 0, 1])
    >>> p_to_q(np.array([2,4,8,3,9,7,6,8,9,9]))
    array([2, 4, 8, 3, 9, 7, 6, 0, 1, 5])
    """
    q = []
    for i in range(len(p)):
        q.append(i)

    i = 0
    for num in p:
        q[i], q[num] = q[num], q[i]
        i += 1

    return np.array(q)


# Q4b
def solve_plu(A, b):
    """
    return the solution of Ax=b. The solution is calculated
    by calling scipy.linalg.lu_factor, converting the piv
    vector using p_to_q, and solving two triangular linear systems
    using scipy.linalg.solve_triangular.
    """
    lu, piv = sla.lu_factor(A)
    q = p_to_q(piv)
    b = b[q]
    y = sla.solve_triangular(lu, b, lower=True, unit_diagonal=True)
    x = sla.solve_triangular(lu, y)
    return x


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # test your solve_plu function on a random system
    n = 10
    A = np.random.uniform(-1, 1, [n, n])
    b = np.random.uniform(-1, 1, n)
    xtrue = sla.solve(A, b)
    x = solve_plu(A, b)
    print("solve_plu works:", np.allclose(x, xtrue, rtol=1e-10, atol=0))
