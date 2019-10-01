import numpy as np
from numpy.polynomial.chebyshev import *
from numpy.polynomial.legendre import legroots
from scipy.special import eval_chebyt, eval_chebyu, roots_chebyt, eval_legendre, roots_legendre
from scipy.linalg import block_diag, lstsq, solve, inv
from cheby import *

class CollocationSolution():
    def __init__(self, N, degree, breakpoints, continuous=False):
        try:
            L = len(continuous)
            self.continuous = continuous
        except:
            self.continuous = [continuous for n in range(N)]
        self.N = N
        self.degree = degree
        self.breakpoints = breakpoints
        self.points, self.weights = roots_legendre(self.degree)
        self.intervals = len(self.breakpoints) - 1
        self.collocation_points = np.concatenate([(self.points + 1) / 2 * (self.breakpoints[i+1] - self.breakpoints[i]) + self.breakpoints[i] for i in range(self.intervals)])
        self.components = [UnivariateCollocationSolution(self.degree, self.breakpoints, self.continuous[n]) for n in range(self.N)]
        self.dimension = sum([self.components[n].dimension for n in range(self.N)])

    def __call__(self, x):
        return self.eval(x)

    def get_coeffs(self):
        return np.concatenate([self.components[n].coeffs for n in range(self.N)])

    def update_coeffs(self, vals):
        start = 0
        for n in range(self.N):
            self.components[n].update_coeffs(vals[start:start+self.components[n].dimension])
            start += self.components[n].dimension

    def fit(self, x, data, degree=None):
        if degree is None:
            degree = self.degree
        for n in range(self.N):
            self.components[n].fit(x, data[n], degree=degree)

    def interpolate(self, fun_list):
        for n in range(self.N):
            self.components[n].interpolate(fun_list[n])

    def fitting_matrix(self):
        return block_diag(*(self.components[n].fitting_matrix() for n in range(self.N)))

    def eval(self, x):
        return np.array([self.components[n].eval(x) for n in range(self.N)])

    def eval_matrix(self, x):
        return block_diag(*(self.components[n].eval_matrix(x) for n in range(self.N)))

    def derivative(self, x):
        return np.array([self.components[n].derivative(x) for n in range(self.N)])

    def derivative_matrix(self, x=None):
        return block_diag(*(self.components[n].derivative_matrix(x=x) for n in range(self.N)))

    def continuity_error(self):
        return np.concatenate([self.components[n].continuity_error() for n in range(self.N)])

    def continuity_jacobian(self):
        cc_jac = np.zeros(((self.intervals-1)*np.count_nonzero(self.continuous), self.dimension))
        row = 0
        col = 0
        for n in range(self.N):
            if self.continuous[n]:
                cc_jac[row:row+self.intervals-1,col:col+self.components[n].dimension] = self.components[n].continuity_jacobian()
                row += self.intervals-1
            col += self.components[n].dimension
        return cc_jac

    def integral(self):
        return np.array([self.components[n].integral() for n in range(self.N)])

    def antiderivative(self, x):
        return np.array([self.components[n].antiderivative(x) for n in range(self.N)])

class UnivariateCollocationSolution():
    def __init__(self, degree, breakpoints, continuous=False):
        self.continuous = continuous
        self.degree = degree + int(self.continuous)
        self.breakpoints = breakpoints
        self.points, self.weights = roots_legendre(self.degree - int(self.continuous))
        self.intervals = len(self.breakpoints) - 1
        self.dimension = self.intervals*self.degree
        self.coeffs = np.zeros(self.dimension, dtype=np.float64)
        self.collocation_points = np.concatenate([(self.points + 1) / 2 * (self.breakpoints[i+1] - self.breakpoints[i]) + self.breakpoints[i] for i in range(self.intervals)])

    def __call__(self, x):
        return self.eval(x)

    def __mul__(self, other):
        if not np.allclose(self.breakpoints, other.breakpoints):
            raise ValueError('Breakpoints must match')
        prod = UnivariateCollocationSolution(self.degree + other.degree, self.breakpoints, self.continuous and other.continuous)
        for i in range(prod.intervals):
            multiplication = chebmul(self.coeffs[i*self.degree:(i+1)*self.degree], other.coeffs[i*other.degree:(i+1)*other.degree])
            prod.coeffs[i*prod.degree:i*prod.degree+multiplication.shape[0]] = multiplication
        return prod

    def update_coeffs(self, coeffs):
        self.coeffs = coeffs

    def fit(self, x, data, degree=None):
        self.coeffs = np.zeros(self.dimension, dtype=np.float64)
        if degree is None:
            degree = self.degree
        for i in range(self.intervals-1):
            mask = np.logical_and(self.breakpoints[i] <= x, x < self.breakpoints[i+1])
            shifted = (x[mask] - self.breakpoints[i]) / (self.breakpoints[i+1] - self.breakpoints[i]) * 2 - 1
            self.coeffs[i*self.degree:i*self.degree+degree] = chebfit(shifted, data[mask], degree-1)

        i = self.intervals - 1
        mask = np.logical_and(self.breakpoints[i] <= x, x <= self.breakpoints[i+1])
        shifted = (x[mask] - self.breakpoints[i]) / (self.breakpoints[i+1] - self.breakpoints[i]) * 2 - 1
        self.coeffs[i*self.degree:i*self.degree+degree] = chebfit(shifted, data[mask], degree-1)

    def interpolate(self, fun):

        self.fit(self.collocation_points, fun(self.collocation_points), degree=self.degree-int(self.continuous))

    def fitting_matrix(self):
        if self.continuous:
            A = np.zeros((self.degree,self.degree-1))
            A[:-1,:] = inv(chebvander(self.points, self.degree-2))
        else:
            A = inv(chebvander(self.points, self.degree-1))
        return block_diag(*(A for i in range(self.intervals)))

    def eval(self, x):
        try:
            K = x.shape[0]
            result = evaluate_piecewise(self.degree, self.coeffs, x, self.breakpoints)
        except:
            x = np.array([x], dtype=np.float64)
            K = 1
            result = evaluate_piecewise(self.degree, self.coeffs, x, self.breakpoints)[0]

        return result

    def eval_matrix(self, x):
        try:
            K = x.shape[0]
        except:
            K = 1
            x = np.array([x])
        return evaluation_matrix(self.degree, x, self.breakpoints)

    def derivative(self, x):
        try:
            K = x.shape[0]
            result = evaluate_derivative(self.degree, self.coeffs, x, self.breakpoints)
        except:
            x = np.array([x], dtype=np.float64)
            K = 1
            result = evaluate_derivative(self.degree, self.coeffs, x, self.breakpoints)[0]

        return result

    def derivative_matrix(self, x=None):
        if x is None:
            D = np.zeros((len(self.points), self.degree))
            for d in range(1, self.degree):
                D[:,d] = d*eval_chebyu(d-1, self.points)
            return block_diag(*(2/(self.breakpoints[i+1] - self.breakpoints[i]) * D for i in range(self.intervals)))
        else:
            try:
                K = len(x)
            except:
                K = 1
                x = np.array([x])
            D = np.zeros((K, self.intervals*self.degree))
            for i in range(self.intervals):
                mask = np.logical_and(self.breakpoints[i] <= x, x < self.breakpoints[i+1])
                shifted = (x[mask] - self.breakpoints[i]) / (self.breakpoints[i+1] - self.breakpoints[i]) * 2 - 1
                for d in range(1, self.degree):
                    D[mask,self.degree*i+d] = d*eval_chebyu(d-1, shifted) * 2/(self.breakpoints[i+1] - self.breakpoints[i])

            i = self.intervals-1
            mask = np.isclose(x, 1.0)
            shifted = (x[mask] - self.breakpoints[i]) / (self.breakpoints[i+1] - self.breakpoints[i]) * 2 - 1
            for d in range(1, self.degree):
                D[mask,self.degree*i+d] = d*eval_chebyu(d-1, shifted) * 2/(self.breakpoints[i+1] - self.breakpoints[i])

            return D

    def continuity_error(self):
        if self.continuous:
            start_values = evaluate_piecewise(self.degree, self.coeffs, self.breakpoints[1:-1], self.breakpoints)
            end_values = np.sum(np.reshape(self.coeffs, (self.degree, self.intervals), order='F'), axis=0)[:-1]
            result = start_values - end_values
        else:
            result = np.array([])
        return result

    def continuity_jacobian(self):
        if self.continuous:
            B0 = self.eval_matrix(self.breakpoints[1:-1])
            B1 = np.zeros(B0.shape)
            B1[:,:-self.degree] = block_diag(*(np.ones(self.degree) for i in range(self.intervals-1)))
            result = B0 - B1
        else:
            result = np.array([])
        return result

    def single_integral(self, coeffs):
        result = coeffs[0] * 2
        for d in range(2, len(coeffs)):
            result += coeffs[d] * ((-1)**d + 1) / (1 - d**2)
        return result

    def integral(self):
        result = 0.0
        for i in range(self.intervals):
            result += (self.breakpoints[i+1] - self.breakpoints[i]) / 2 * self.single_integral(self.coeffs[i*self.degree:(i+1)*self.degree])
        return result

    def antiderivative(self, x):
        try:
            K = x.shape[0]
            result = evaluate_antiderivative(self.degree, self.coeffs, x, self.breakpoints)
        except:
            K = 1
            x = np.array([x])
            result = evaluate_antiderivative(self.degree, self.coeffs, x, self.breakpoints)[0]
        return result

    def deriv(self):
        deriv = UnivariateCollocationSolution(self.degree-1, self.breakpoints, False)
        for i in range(deriv.intervals):
            deriv.coeffs[i*deriv.degree:(i+1)*deriv.degree] = chebder(self.coeffs[i*self.degree:(i+1)*self.degree]) / (self.breakpoints[i+1] - self.breakpoints[i]) * 2
        return deriv
