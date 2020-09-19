#CSC336 Assignment #1 starter code for the report question
import numpy as np
import matplotlib.pyplot as plt

"""
See the examples in class this week if you
aren't sure how to start writing the code
to run the experiment and produce the plot for Q1.

We won't be grading your code unless there is something
strange with your output, so don't worry too much about
coding style.

Please start early and ask questions.

A few things you'll likely find useful:

import matplotlib.pyplot as plt

hs = np.logspace(-15,-1,15)
plt.figure()
plt.loglog(hs,rel_errs)
plt.show() #displays the figure
plt.savefig("myplot.png")



a_cmplx_number = 1j #the j in python is the complex "i"

try to reuse code where possible ("2 or more, use a for")

"""


#example function header, you don't have to use this
def fd(f, x, h):
    """
    Return the forward finite difference approximation
    to the derivative of f(x), using the step size(s)
    in h.

    f is a function

    x is a scalar

    h may be a scalar or a numpy array (like the output of
    np.logspace(-15,-1,15) )

    """
    to_ret = []
    for difference in h:
        top = np.subtract(f(np.add(x, difference)), f(x))
        to_ret.append(np.divide(top, difference))
    return np.array(to_ret)


def centred_difference(f, x, h):
    """
    Returns the centred finite difference approximation
    to the derivative of f(x), using a step size(s) in h
    """
    to_ret = []
    for difference in h:
        top = np.subtract(f(np.add(x, difference)),
                          f(np.subtract(x, difference)))
        to_ret.append(np.divide(top, 2 * difference))
    return np.array(to_ret)


def complex_step(f, x, h):
    """
    Returns the complex step approximation to the
    derivative of f(x), using a step size(s) in h
    """
    to_ret = []
    for difference in h:
        z = complex(x, difference)
        # to_ret.append(np.divide(f(z).imag, difference))
        to_ret.append(f(z).imag / difference)
    return np.array(to_ret)


def experiment_function(x):
    """
    f(x) = e^x / sqrt(sin^3(x) + cos^3(x))
    """
    top = np.exp(x)
    bottom = np.sqrt(np.sin(x) ** 3 + np.cos(x) ** 3)
    return np.divide(top, bottom)


if __name__ == "__main__":
    hs = np.logspace(-15, -1, 15)
    answer = 4.05342789389862
    results = fd(experiment_function, 1.5, hs)
    rele_err = abs(np.divide(np.subtract(results, answer), answer))
    plt.figure()
    plt.loglog(hs, rele_err)
    results = centred_difference(experiment_function, 1.5, hs)
    rele_err = abs(np.divide(np.subtract(results, answer), answer))
    plt.loglog(hs, rele_err)
    results = complex_step(experiment_function, 1.5, hs)
    rele_err = abs(np.divide(np.subtract(results, answer), answer))
    plt.loglog(hs, rele_err)
    plt.legend(['Forward difference', 'Centred difference',
                'Complex step method'])
    plt.show()  # displays the figure