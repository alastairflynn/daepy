import numpy as np
cimport cython
from cython.parallel cimport prange, parallel
from scipy.special.cython_special cimport eval_chebyt, eval_chebyu

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double _evaluate_single(const int degree, const double[:] coef, const double x, const double[:] breakpoints) nogil:
    cdef int n, i, start
    cdef int intervals = breakpoints.shape[0] - 1
    cdef double out = 0.0
    cdef double sign, shifted

    i = 0
    while x >= breakpoints[i+1]:
        if i == intervals-1:
            break
        else:
            i = i+1
    start = i*degree
    shifted = (x - breakpoints[i]) / (breakpoints[i+1] - breakpoints[i]) * 2.0 - 1.0

    for n in range(degree):
        out = out + coef[start+n]*eval_chebyt(n, shifted)

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _evaluate_piecewise(const int degree, const double[:] coef, const double[:] x, const double[:] breakpoints, double[:] out) nogil:
    cdef int k
    for k in prange(x.shape[0]):
        out[k] = _evaluate_single(degree, coef, x[k], breakpoints)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double _derivative_single(const int degree, const double[:] coef, const double x, const double[:] breakpoints) nogil:
    cdef int n, i, start
    cdef int intervals = breakpoints.shape[0] - 1
    cdef double out = 0.0
    cdef double sign, shifted

    i = 0
    while x >= breakpoints[i+1]:
        if i == intervals-1:
            break
        else:
            i = i+1
    start = i*degree
    shifted = (x - breakpoints[i]) / (breakpoints[i+1] - breakpoints[i]) * 2.0 - 1.0

    for n in range(1,degree):
        out = out + n*coef[start+n]*eval_chebyu(n-1, shifted)

    return out / (breakpoints[i+1] - breakpoints[i]) * 2.0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _evaluate_derivative(const int degree, const double[:] coef, const double[:] x, const double[:] breakpoints, double[:] out) nogil:
    cdef int k
    for k in prange(x.shape[0]):
        out[k] = _derivative_single(degree, coef, x[k], breakpoints)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double _antiderivative_single_interval(const int degree, const double[:] coef, const double x) nogil:
    cdef int n
    cdef double out = 0.0
    cdef double c = 0.0

    out = coef[0] * x
    c = -coef[0]
    if degree >= 2:
        out = out + coef[1] * x**2 / 2.0
        c = c + coef[1] / 2.0
    for n in range(2,degree):
        out = out + coef[n] * (eval_chebyt(n+1, x) / (n+1) - eval_chebyt(n-1, x) / (n-1)) / 2.0
        c = c + coef[n] * ((-1.0)**(n+1) / (n+1) - (-1.0)**(n-1) / (n-1)) / 2.0
    return out - c

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double _antiderivative_single(const int degree, const double[:] coef, const double x, const double[:] breakpoints) nogil:
    cdef int n, i, start, end
    cdef int intervals = breakpoints.shape[0] - 1
    cdef double shifted, c

    i = 0
    c = 0.0
    while x >= breakpoints[i+1]:
        if i == intervals-1:
            break
        else:
            start = i*degree
            end = (i+1)*degree
            c = c + _antiderivative_single_interval(degree, coef[start:end], 1.0) * (breakpoints[i+1] - breakpoints[i]) / 2.0
            i = i+1
    start = i*degree
    end = (i+1)*degree
    shifted = (x - breakpoints[i]) / (breakpoints[i+1] - breakpoints[i]) * 2.0 - 1.0
    return c + _antiderivative_single_interval(degree, coef[start:end], shifted) * (breakpoints[i+1] - breakpoints[i]) / 2.0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _evaluate_antiderivative(const int degree, const double[:] coef, const double[:] x, const double[:] breakpoints, double[:] out) nogil:
    cdef int k
    for k in prange(x.shape[0]):
        out[k] = _antiderivative_single(degree, coef, x[k], breakpoints)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _vandermode_single(const int degree, const double x, const double[:] breakpoints, double[:] work) nogil:
    cdef int n, i, start
    cdef int intervals = breakpoints.shape[0] - 1
    cdef double shifted, sign

    i = 0
    while x >= breakpoints[i+1]:
        if i == intervals-1:
            break
        else:
            i = i+1
    start = i*degree
    shifted = (x - breakpoints[i]) / (breakpoints[i+1] - breakpoints[i]) * 2.0 - 1.0

    for n in range(degree):
        work[start+n] = eval_chebyt(n, shifted)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _evaluation_matrix(const int degree, const double[:] x, const double[:] breakpoints, double[:,:] A):
    cdef int k
    with nogil, parallel():
        for k in prange(x.shape[0]):
            _vandermode_single(degree, x[k], breakpoints, A[k])

def evaluate_piecewise(degree, coef, x, breakpoints):
    out = np.empty_like(x, dtype='float64')
    _evaluate_piecewise(degree, coef, x, breakpoints, out)
    return out

def evaluate_derivative(degree, coef, x, breakpoints):
    out = np.empty_like(x, dtype='float64')
    _evaluate_derivative(degree, coef, x, breakpoints, out)
    return out

def evaluate_antiderivative(degree, coef, x, breakpoints):
    out = np.empty_like(x, dtype='float64')
    _evaluate_antiderivative(degree, coef, x, breakpoints, out)
    return out

def evaluation_matrix(degree, x, breakpoints):
    rows = x.shape[0]
    cols = degree*(len(breakpoints)-1)
    A = np.zeros((rows, cols))
    _evaluation_matrix(degree, x, breakpoints, A)
    return A
