
"""
Demonstration module for quadratic interpolation.
Update this docstring to describe your code.
Modified by: ** Nitin Singh **
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

def quad_interp(xi,yi):
    """
    Quadratic interpolation.  Compute the coefficients of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2.
    Returns c, an array containing the coefficients of
      p(x) = c[0] + c[1]*x + c[2]*x**2.

    """

    # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length 3"
    assert len(xi)==3 and len(yi)==3, error_message

    # Set up linear system to interpolate through data points:
    A = np.vstack([np.ones(3), xi, xi**2]).T
    b = yi
    
    ### Fill in this part to compute c ###
    c = solve(A,b)

    return c


def plot_quad(xi,yi):
    """
    Plots the interpolating polynomial & data points
    and save the fig as quadratic.png
    """
    # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length 3"
    assert len(xi)==3 and len(yi)==3, error_message

    # Get c, an array containing the coefficients of
    # p(x) = c[0] + c[1]*x + c[2]*x**2.
    c = quad_interp(xi,yi)

    # Plot the resulting polynomial:
    x = np.linspace(xi.min() - 1,  xi.max() + 1, 1000) # points to evaluate polynomial
    y = c[0] + c[1]*x + c[2]*x**2

    plt.figure(1)       # open plot figure window
    plt.clf()           # clear figure
    plt.plot(x,y,'b-')  # connect points with a blue line

    # Add data points  (polynomial should go through these points!)
    plt.plot(xi,yi,'ro')   # plot as red circles
    plt.ylim(yi.min() - 1, yi.max() + 1) # set limits in y for plot
    # plt.ylim(-2,8)         # set limits in y for plot

    plt.title("Data points and interpolating polynomial")

    plt.savefig('quadratic.png')   # save figure as .png file
    
def test_quad1():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  2.])
    yi = np.array([ 1., -1.,  7.])
    c = quad_interp(xi,yi)
    c_true = np.array([-1.,  0.,  2.])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)


def test_quad2():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-2.,  1.,  5.])
    yi = np.array([ 7., 1.,  49.])
    c = quad_interp(xi,yi)
    c_true = np.array([-1.,  0.,  2.])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)

def cubic_interp(xi,yi):
    """
    Cubic interpolation.  Compute the coefficients of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2,3.
    Returns c, an array containing the coefficients of
      p(x) = c[0] + c[1]*x + c[2]*x**2 + c[3]*x^3.

    """

    # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length 4"
    assert len(xi)==4 and len(yi)==4, error_message

    # Set up linear system to interpolate through data points:
    A = np.vstack([np.ones(4), xi, xi**2, xi**3]).T
    b = yi
    
    ### Fill in this part to compute c ###
    c = solve(A,b)

    return c

def plot_cubic(xi,yi):
    """
    Plots the interpolating polynomial & data points
    and save the fig as quadratic.png
    """
    # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length 4"
    assert len(xi)==4 and len(yi)==4, error_message
    
    # Get c, an array containing the coefficients of
    # p(x) = c[0] + c[1]*x + c[2]*x**2 + c[3]*x^3.
    c = cubic_interp(xi,yi)

    # Plot the resulting polynomial:
    x = np.linspace(xi.min() - 1,  xi.max() + 1, 1000) # points to evaluate polynomial
    y = c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3

    plt.figure(1)       # open plot figure window
    plt.clf()           # clear figure
    plt.plot(x,y,'b-')  # connect points with a blue line

    # Add data points  (polynomial should go through these points!)
    plt.plot(xi,yi,'ro')   # plot as red circles
    plt.ylim(-2,100)         # set limits in y for plot

    plt.title("Data points and interpolating polynomial")

    plt.savefig('cubic.png')   # save figure as .png file
    
def test_cubic1():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  2., 4.])
    yi = np.array([ 0., -1.,  15., 95.])    
    c = cubic_interp(xi,yi)
    c_true = np.array([-1.,  0.,  2., 1.])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)

def poly_interp(xi,yi):
    """
    Poly interpolation. Compute the coefficients of the polynomial
    interpolating the points (xi[i],yi[i]) for i = 0,1,2,3.. n-1.
    Returns c, an array containing the coefficients of
      p(x) = c[0] + c[1]*x + c[2]*x**2 ... + c[n-1]*x**(n-1)

    """

    # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have same length"
    assert len(xi)==len(yi), error_message

    # Set up linear system to interpolate through data points:
    A = np.vstack([np.ones(len(xi))])

    for i in range(1, len(xi)):
        row = xi**(i)
        #print "Row is: ", row
        #row = np.array([xi**(i)])
        A = np.append(A, [row], axis=0)
        #print "A is: ", A
    
    #A = np.vstack([np.ones(3), xi, xi**2]).T
    A = A.T
    b = yi

    ### Fill in this part to compute c ###
    c = solve(A,b)

    return c


def plot_poly(xi,yi):
    """
    Plots the interpolating polynomial & data points
    and save the fig as quadratic.png
    """
    # check inputs and print error message if not valid:

    error_message = "xi and yi should have type numpy.ndarray"
    assert (type(xi) is np.ndarray) and (type(yi) is np.ndarray), error_message

    error_message = "xi and yi should have length 3"
    assert len(xi)==len(yi), error_message

    # Get c, an array containing the coefficients of
    # p(x) = c[0] + c[1]*x + c[2]*x**2 ... + c[n-1]*x**(n-1)
    c = poly_interp(xi,yi)

    # Plot the resulting polynomial:
    x = np.linspace(xi.min() - 1,  xi.max() + 1, 1000) # points to evaluate polynomial
    #y = c[0] + c[1]*x + c[2]*x**2
    n = len(xi)
    y = c[n-1]
    for j in range(n-1, 0, -1):
        y = y*x + c[j-1]

    plt.figure(1)       # open plot figure window
    plt.clf()           # clear figure
    plt.plot(x,y,'b-')  # connect points with a blue line

    # Add data points  (polynomial should go through these points!)
    plt.plot(xi,yi,'ro')   # plot as red circles
    plt.ylim(yi.min() - 1, yi.max() + 10)         # set limits in y for plot

    plt.title("Data points and interpolating polynomial")

    plt.savefig('poly.png')   # save figure as .png file

def test_poly1():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-1.,  0.,  2., 4.])
    yi = np.array([ 0., -1.,  15., 95.])
    c = poly_interp(xi,yi)
    c_true = np.array([-1.,  0.,  2., 1.])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)
    plot_poly(xi,yi)

def test_poly2():
    """
    Test code, no return value or exception if test runs properly.
    """
    xi = np.array([-3., -1.,  0.,  2., 4.])
    yi = np.array([ 152., 2., -1., 47., 607.])    
    c = poly_interp(xi,yi)
    c_true = np.array([-1.,  0.,  2., 1., 2.])
    print "c =      ", c
    print "c_true = ", c_true
    # test that all elements have small error:
    assert np.allclose(c, c_true), \
        "Incorrect result, c = %s, Expected: c = %s" % (c,c_true)
    #plot_poly(xi,yi)
    
    
if __name__=="__main__":
    # "main program"
    # the code below is executed only if the module is executed at the command line,
    #    $ python demo2.py
    # or run from within Python, e.g. in IPython with
    #    In[ ]:  run demo2
    # not if the module is imported.
    print "Running test quad1 ..."
    test_quad1()
    print "Running test quad2 ..."
    test_quad2()
    print "Running test cubic1 ..."
    test_cubic1()
    print "Running test poly1 ..."
    test_poly1()
    print "Running test poly21 ..."
    test_poly2()
