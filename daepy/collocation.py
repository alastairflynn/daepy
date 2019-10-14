import numpy as np
from numpy.polynomial.chebyshev import chebmul, chebfit, chebvander, chebder
from scipy.special import eval_chebyu, roots_legendre
from scipy.linalg import block_diag, inv
from .cheby import evaluate_piecewise, evaluation_matrix, evaluate_derivative, evaluate_antiderivative

class CollocationSolution():
    '''
    Multivariate piecewise polynomial where *N* is the number of components, *degree* is the degree of the continuous components, *breakpoints* are the ends of the subintervals and *continuous* is an (N,) numpy array of booleans determining which components are continuous. If *continuous* is `True` or `False` then all components are set to be continuous or not respectively.

    Each component is a :class:`UnivariateCollocationSolution`. These can be accessed by indexing the object or via the :attr:`components` attribute which is a list of the :class:`UnivariateCollocationSolution` components.

    .. note::
        Labelling a component as continuous does not guarantee that it will be continuous, it only means that it will be represented by poynomials one degree higher than components labelled as not continuous.
    '''
    def __init__(self, N, degree, breakpoints, continuous=False):
        try:
            L = len(continuous)
            self.continuous = continuous
        except TypeError:
            self.continuous = [continuous for n in range(N)]
        self.N = N
        self.degree = degree
        self.breakpoints = breakpoints
        self.points, self.weights = roots_legendre(self.degree)
        self.intervals = len(self.breakpoints) - 1
        self.collocation_points = np.concatenate([(self.points + 1) / 2 * (self.breakpoints[i+1] - self.breakpoints[i]) + self.breakpoints[i] for i in range(self.intervals)])
        self.components = [UnivariateCollocationSolution(self.degree, self.breakpoints, self.continuous[n]) for n in range(self.N)]
        self.dimension = sum([self[n].dimension for n in range(self.N)])

    def __call__(self, x):
        return self.eval(x)

    def __getitem__(self, n):
        return self.components[n]

    def get_coeffs(self):
        '''
        Returns the polynomial coefficients.
        '''
        return np.concatenate([self[n].coeffs for n in range(self.N)])

    def update_coeffs(self, vals):
        '''
        Update the polynomial coefficients.
        '''
        start = 0
        for n in range(self.N):
            self[n].update_coeffs(vals[start:start+self[n].dimension])
            start += self[n].dimension

    def fit(self, x, data, degree=None):
        '''
        Fit the polynomials to an (N,K) numpy array *data* at points given by (K,) numpy array *x*. Optionally limit the fitting to degree *degree*.
        '''
        if degree is None:
            degree = self.degree
        for n in range(self.N):
            self[n].fit(x, data[n], degree=degree)

    def interpolate(self, fun_list):
        '''
        Interpolate a list of functions.
        '''
        for n in range(self.N):
            self[n].interpolate(fun_list[n])

    def fitting_matrix(self):
        '''
        Return the matrix :math:`F` such that :math:`c = Fd` are the polynomial coefficients where :math:`d` is fitting data.
        '''
        return block_diag(*(self[n].fitting_matrix() for n in range(self.N)))

    def eval(self, x):
        '''
        Evaluate the piecewise polynomials. This can also be achieved by simply calling the object like a function, that is :code:`sol(x)` is equivalent to :code:`sol.eval(x)`.
        '''
        result = np.array([self[n].eval(x) for n in range(self.N)])
        if self.N == 1:
            result = np.reshape(result, -1)
        return result

    def eval_matrix(self, x):
        '''
        Return the matrix :math:`E` such that :math:`Ec = y(x)` where :math:`c` are the polynomial coefficients and :math:`y(x)` is the piecewise polynomial evaluated at points *x*.
        '''
        return block_diag(*(self[n].eval_matrix(x) for n in range(self.N)))

    def derivative(self, x):
        '''
        Calculate the derivative at points *x*.
        '''
        return np.array([self[n].derivative(x) for n in range(self.N)])

    def derivative_matrix(self, x=None):
        '''
        Return the matrix :math:`D` such that :math:`Dc = y'(x)` where :math:`c` are the polynomial coefficients and :math:`y'(x)` is the derivative of the piecewise polynomial evaluated at points *x*. If *x* is not given then it is taken to be the collocation points and the matrix is constructed using a faster routine than for general *x*.
        '''
        return block_diag(*(self[n].derivative_matrix(x) for n in range(self.N)))

    def continuity_error(self):
        '''
        Return the continuity error for continuous variables.
        '''
        return np.concatenate([self[n].continuity_error() for n in range(self.N)])

    def continuity_jacobian(self):
        '''
        Return the jacobian of the continuity error for continuous variables.
        '''
        cc_jac = np.zeros(((self.intervals-1)*np.count_nonzero(self.continuous), self.dimension))
        row = 0
        col = 0
        for n in range(self.N):
            if self.continuous[n]:
                cc_jac[row:row+self.intervals-1,col:col+self[n].dimension] = self[n].continuity_jacobian()
                row += self.intervals-1
            col += self[n].dimension
        return cc_jac

    def integral(self):
        '''
        Integrate the piecewise polynomial over the whole interval.
        '''
        return np.array([self[n].integral() for n in range(self.N)])

    def antiderivative(self, x):
        '''
        Calculate the antiderivative of the piecewise polynomial at points *x*. The antiderivative at 0 is 0.
        '''
        return np.array([self[n].antiderivative(x) for n in range(self.N)])

class UnivariateCollocationSolution():
    def __init__(self, degree, breakpoints, continuous=False):
        '''
        Piecewise polynomial where *degree* is the degree of the pieces, *breakpoints* are the ends of the subintervals and *continuous* is `True` or `False`. If *continuous* is `False` then the pieces are one degree less than *degree*.

        ..note::Setting continuous to be `True` does not guarantee that the piecewise polynomial will be continuous.
        '''
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
        '''
        Update the polynomial coefficients.
        '''
        self.coeffs = coeffs

    def fit(self, x, data, degree=None):
        '''
        Fit the polynomials to an (K,) numpy array *data* at points given by (K,) numpy array *x*. Optionally limit the fitting to degree *degree*.
        '''
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
        '''
        Interpolate a function.
        '''
        data = fun(self.collocation_points)
        try:
            K = data.shape[0]
        except AttributeError:
            data = data*np.ones(self.collocation_points.shape[0])
        self.fit(self.collocation_points, data, degree=self.degree-int(self.continuous))

    def fitting_matrix(self):
        '''
        Return the matrix :math:`F` such that :math:`c = Fd` are the polynomial coefficients where :math:`d` is fitting data.
        '''
        if self.continuous:
            A = np.zeros((self.degree,self.degree-1))
            A[:-1,:] = inv(chebvander(self.points, self.degree-2))
        else:
            A = inv(chebvander(self.points, self.degree-1))
        return block_diag(*(A for i in range(self.intervals)))

    def eval(self, x):
        '''
        Evaluate the piecewise polynomials. This can also be achieved by simply calling the object like a function, that is :code:`sol(x)` is equivalent to :code:`sol.eval(x)`.
        '''
        try:
            K = x.shape[0]
            result = evaluate_piecewise(self.degree, self.coeffs, x, self.breakpoints)
        except (IndexError, AttributeError):
            x = np.array([x], dtype=np.float64)
            K = 1
            result = evaluate_piecewise(self.degree, self.coeffs, x, self.breakpoints)[0]

        return result

    def eval_matrix(self, x):
        '''
        Return the matrix :math:`E` such that :math:`Ec = y(x)` where :math:`c` are the polynomial coefficients and :math:`y(x)` is the piecewise polynomial evaluated at points *x*.
        '''
        try:
            K = x.shape[0]
        except:
            K = 1
            x = np.array([x])
        return evaluation_matrix(self.degree, x, self.breakpoints)

    def derivative(self, x):
        '''
        Calculate the derivative at points *x*.
        '''
        try:
            K = x.shape[0]
            result = evaluate_derivative(self.degree, self.coeffs, x, self.breakpoints)
        except:
            x = np.array([x], dtype=np.float64)
            K = 1
            result = evaluate_derivative(self.degree, self.coeffs, x, self.breakpoints)[0]

        return result

    def derivative_matrix(self, x=None):
        '''
        Return the matrix :math:`D` such that :math:`Dc = y'(x)` where :math:`c` are the polynomial coefficients and :math:`y'(x)` is the derivative of the piecewise polynomial evaluated at points *x*. If *x* is not given then it is taken to be the collocation points and the matrix is constructed using a faster routine than for general *x*.
        '''
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
        '''
        Return the continuity error for if continuous, otherwise return `None`.
        '''
        if self.continuous:
            start_values = evaluate_piecewise(self.degree, self.coeffs, self.breakpoints[1:-1], self.breakpoints)
            end_values = np.sum(np.reshape(self.coeffs, (self.degree, self.intervals), order='F'), axis=0)[:-1]
            result = start_values - end_values
        else:
            result = np.array([])
        return result

    def continuity_jacobian(self):
        '''
        Return the jacobian of the continuity error if continuous, otherwise return an empty numpy array.
        '''
        if self.continuous:
            B0 = self.eval_matrix(self.breakpoints[1:-1])
            B1 = np.zeros(B0.shape)
            B1[:,:-self.degree] = block_diag(*(np.ones(self.degree) for i in range(self.intervals-1)))
            result = B0 - B1
        else:
            result = np.array([])
        return result

    def single_integral(self, coeffs):
        '''
        Integrate a single polynomial with coefficients *coeffs*. To integrate the piecewise polynomial over the whole interval use :meth:`integral`.
        '''
        result = coeffs[0] * 2
        for d in range(2, len(coeffs)):
            result += coeffs[d] * ((-1)**d + 1) / (1 - d**2)
        return result

    def integral(self):
        '''
        Integrate the piecewise polynomial over the whole interval.
        '''
        result = 0.0
        for i in range(self.intervals):
            result += (self.breakpoints[i+1] - self.breakpoints[i]) / 2 * self.single_integral(self.coeffs[i*self.degree:(i+1)*self.degree])
        return result

    def antiderivative(self, x):
        '''
        Calculate the antiderivative of the piecewise polynomial at points *x*. The antiderivative at 0 is 0.
        '''
        try:
            K = x.shape[0]
            result = evaluate_antiderivative(self.degree, self.coeffs, x, self.breakpoints)
        except:
            K = 1
            x = np.array([x])
            result = evaluate_antiderivative(self.degree, self.coeffs, x, self.breakpoints)[0]
        return result

    def deriv(self):
        '''
        Return the derivative as a new :class:`UnivariateCollocationSolution`. If you merely want to evaluate the derivative at a set of points then it is faster to use :meth:`derivative`.
        '''
        deriv = UnivariateCollocationSolution(self.degree-1, self.breakpoints, False)
        for i in range(deriv.intervals):
            deriv.coeffs[i*deriv.degree:(i+1)*deriv.degree] = chebder(self.coeffs[i*self.degree:(i+1)*self.degree]) / (self.breakpoints[i+1] - self.breakpoints[i]) * 2
        return deriv
