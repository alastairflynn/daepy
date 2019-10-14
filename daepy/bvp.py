import numpy as np
from scipy.sparse import bmat, csc_matrix, csr_matrix, diags
import dill
from multiprocessing import Pool
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from .collocation import CollocationSolution
from .nonlinear import fsolve
from .continuation import BVPContinuation
from .derivatives import approx_jac

class BVP():
    '''
    This class is used to construct and solve the nonlinear system. It is initialised with a class *dae* which follows the :class:`.DAETemplate`, the *degree* of the differential variables and the number of *intervals* in the mesh.
    '''
    def __init__(self, dae, degree=3, intervals=10):
        self.dae = dae
        continuous = np.array([False for n in range(dae.N)])
        continuous[dae.dindex] = True
        self.bvpsol = BVPSolution(dae.N, degree, intervals, continuous)

        self.collocation_points = self.bvpsol.collocation_points
        self.K = self.collocation_points.shape[0]
        self.dimension = self.bvpsol.dimension + 1

    def solve(self, method='nleqres', tol=1e-8, maxiter=100, disp=False):
        '''
        Solve the nonlinear system where *method* is one of

        * `'nleqres'` a damped global Newton method [1]_ (the default)
        * `'lm'` the Leveberg--Marquardt method [2]_
        * `'partial_inverse'` Newton-like method which calculates a partial inverse of the Jacobian by calculating a QR decomposition and doing a partial backwards substitution when the step doesn’t converge
        * `'steepest_descent'` steepest descent method

        *tol* is required residual tolerance, *maxiter* is the maximum number of iterations and *disp* controls whether convergence messages are printed.

        Returns the final solution as a :class:`BVPSolution`.

        .. [1] P. Deuflhard. Systems of Equations: Global Newton Methods. In *Newton Methods for Nonlinear Problems*, Springer Series in Computational Mathematics, pages 109–172. Springer, Berlin, Heidelberg, 2011.
        .. [2] J. Dennis and R. Schnabel. *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*. Classics in Applied Mathematics. Society for Industrial and Applied Mathematics, January 1996.
        '''
        c,m = fsolve(self.eval, self.state(), self.jac, method, tol, maxiter, disp)
        self.bvpsol.update_coeffs(c)
        return self.bvpsol

    def continuation(self, param, method='pseudo_arclength', steps=1, stepsize=1.0, target=None, tol=1e-8, maxiter=100, disp=False, callback=None):
        '''
        Perform a continuation run starting from parameter value *param* where method is one of

        * `'pseudo_arclength'` the pseudo-arclength method [3]_ (the default)
        * `'naive'` naive continuation with no predictive step

        *steps* is either a maximum number of steps or a numpy array of parameter values which determine the steps explicitly, *stepsize* is the initial stepsize (the pseudo-arclength method will adapt the stepsize, ignored if steps are given explicitly), *target* is a value for the parameter at which the continuation will stop (optional), *tol* is the required solution tolerance, *maxiter* is the maximum number of iterations for the nonlinear solver, *disp* determines whether to print progress messages and *callback(parameter, solution)* is a function that will be called before each continuation step, for example to draw a new line on a plot at each step (optional).

        Returns the final solution as a :class:`BVPSolution`.

        .. note::
            The *bvp* must have been initialised with a *dae* object that defines the :meth:`.update_parameter` method and all the jacobian methods. In the future, continuation using a finite difference approximation of the jacobian may be supported although it would still be strongly recommended to use an analytic jacobian, if available.

        .. [3] E. Allgower and K. Georg. *Introduction to Numerical Continuation Methods*. Classics in Applied Mathematics. Society for Industrial and Applied Mathematics, January 2003.
        '''
        cont = BVPContinuation(self, method)
        coeffs, param = cont.continuation_run(self.state(), param, steps, stepsize, target, tol, maxiter, disp, callback)
        self.bvpsol.update_coeffs(coeffs)
        return self.bvpsol

    def state(self):
        '''
        Returns an array containing the current coefficients of the collocation solution, coordinate transform and the coordinate scaling.
        '''
        return self.bvpsol.state()

    def eval(self, coeffs):
        '''
        Update the polynomial coefficients and evaluate the nonlinear system.
        '''
        self.bvpsol.update_coeffs(coeffs)

        residual = self.dae.fun(self.collocation_points, self.bvpsol).reshape(-1)
        bv_res = self.dae.bv(self.bvpsol).reshape(-1)

        tres0 = self.monitor(self.collocation_points) - self.bvpsol.transform.components[0].derivative(self.collocation_points)
        tres1 = self.bvpsol.transform.components[0](self.bvpsol.transform.components[1](self.collocation_points)) / self.bvpsol.transform.components[0](1) - self.collocation_points

        cc = self.bvpsol.solution.continuity_error()
        cc_transform = self.bvpsol.transform.continuity_error()

        res = np.concatenate([residual, tres0, tres1, cc, cc_transform, bv_res])

        return res

    def jac(self, coeffs):
        '''
        Evaluate the jacobian of the system.
        '''
        self.bvpsol.update_coeffs(coeffs)

        jac, transform_jac = self.dae.jacobian(self.collocation_points, self.bvpsol)
        bv_jac, bv_transform_jac = self.dae.bv_jacobian(self.bvpsol)

        t_jac, t_transform_jac, t_scale_jac = self.monitor_derivative(self.collocation_points)

        cc_jac = self.bvpsol.solution.continuity_jacobian()
        cc_transform_jac = self.bvpsol.transform.continuity_jacobian()

        J = bmat([[jac, transform_jac, None], [t_jac, t_transform_jac, t_scale_jac[:,None]], [cc_jac, None, None], [None, cc_transform_jac, None], [bv_jac, bv_transform_jac, None]], format='csc')

        J.eliminate_zeros()

        return J

    def param_jac(self, x, y):
        '''
        Construct the derivative of the nonlinear system with respect to a parameter.
        '''
        J = np.zeros(self.dimension)
        jac, bv_jac = self.dae.parameter_jacobian(x, y)
        jac = jac.reshape(-1)

        J[:jac.shape[0]] = jac
        J[-bv_jac.shape[0]:] = bv_jac

        return J

    def check_jacobian(self):
        '''
        Draws a plot to show the error between the analytic jacobian and a finite difference jacobian. Can be useful for checking jacobians.
        '''
        analytic = self.jac(self.state())
        stream = dill.dumps(self.eval)
        with Pool() as pool:
            finite_differences = approx_jac(stream, self.state(), 1e-8, pool)

        residual = self.dae.fun(self.collocation_points, self.bvpsol).reshape(-1)
        bv_res = self.dae.bv(self.bvpsol).reshape(-1)
        tres0 = self.monitor(self.collocation_points) - self.bvpsol.transform.components[0].derivative(self.collocation_points)
        tres1 = self.bvpsol.transform.components[0](self.bvpsol.transform.components[1](self.collocation_points)) / self.bvpsol.transform.components[0](1) - self.collocation_points
        cc = self.bvpsol.solution.continuity_error()
        cc_transform = self.bvpsol.transform.continuity_error()

        lines = [residual.shape[0]-0.5, tres0.shape[0] + tres1.shape[0] + residual.shape[0]-0.5, cc.shape[0] + cc_transform.shape[0] + tres0.shape[0] + tres1.shape[0] + residual.shape[0]-0.5]

        error = analytic.toarray() - finite_differences
        print('Maximum jacobian error:', np.max(np.abs(error)))

        fig = plt.figure(figsize=(8,3))
        grid = ImageGrid(fig, 111, nrows_ncols=(1,3), axes_pad=0.5, share_all=True, cbar_location="right", cbar_mode="single", cbar_pad=0.1)

        grid[0].set_title('Analytic jacobian')
        grid[0].imshow(analytic.toarray())
        grid[0].hlines(lines, 0, analytic.shape[1]-1, linestyles='--')
        grid[0].vlines(self.bvpsol.solution.dimension-0.5, 0, analytic.shape[0]-1, linestyles='--')

        grid[1].set_title('Finite difference jacobian')
        grid[1].imshow(finite_differences)
        grid[1].hlines(lines, 0, analytic.shape[1]-1, linestyles='--')
        grid[1].vlines(self.bvpsol.solution.dimension-0.5, 0, analytic.shape[0]-1, linestyles='--')

        grid[2].set_title('Error')
        c = grid[2].imshow(error)
        grid[2].hlines(lines, 0, analytic.shape[1]-1, linestyles='--')
        grid[2].vlines(self.bvpsol.solution.dimension-0.5, 0, analytic.shape[0]-1, linestyles='--')

        grid[2].cax.colorbar(c)
        grid[2].cax.toggle_label(True)

        # plt.tight_layout()
        plt.show()

    def monitor(self, x):
        '''
        The monitor function used to define the coordinate transform.
        '''
        dx = np.sum(self.bvpsol.solution.derivative(x)[self.dae.dindex]**2, axis=0)
        return self.bvpsol.scale / np.sqrt(1 + dx)

    def monitor_derivative(self, x):
        '''
        Derivative of the monitor function.
        '''
        y = self.bvpsol.solution
        y_prime = y.derivative(x)
        dx = np.sum(y_prime[self.dae.dindex]**2, axis=0)

        Jj = np.zeros((2*self.K,self.dae.N*self.K))

        for n in self.dae.dindex:
            Jj[:self.K,n*self.K:(n+1)*self.K] = np.diag(-self.bvpsol.scale*y_prime[n] / np.sqrt(1 + dx)**3)
        M = csc_matrix(y.derivative_matrix())
        jac = csr_matrix(Jj).dot(M)

        Jj = np.zeros((2*self.K, 2*self.K))
        Jj[:self.K,:self.K] = -np.eye(self.K)

        M = csc_matrix(self.bvpsol.transform.derivative_matrix())
        transform_jac = csr_matrix(Jj).dot(M)

        sigma = self.bvpsol.transform.components[1](x)

        Jj = np.zeros((2*self.K, 2*self.K))
        Jj[self.K:,:self.K] = np.eye(self.K) / self.bvpsol.transform.components[0](1)
        derivative_wrt_sigma = np.concatenate([np.diag(self.bvpsol.transform.components[n].derivative(sigma)) for n in range(2)], axis=0)
        sigma_prime = np.zeros((self.K, self.bvpsol.transform.dimension))
        sigma_prime[:,-self.bvpsol.transform.components[1].dimension:] = self.bvpsol.transform.components[1].eval_matrix(x)
        M = csr_matrix(derivative_wrt_sigma).dot(csc_matrix(sigma_prime))
        M = M + csr_matrix(self.bvpsol.transform.eval_matrix(sigma))
        transform_jac += csr_matrix(Jj).dot(M)

        Jj = np.zeros((2*self.K, 2))
        Jj[self.K:,0] = -self.bvpsol.transform.components[0](self.bvpsol.transform.components[1](self.collocation_points)) / self.bvpsol.transform.components[0](1)**2
        M = csr_matrix(self.bvpsol.transform.eval_matrix(1.0))
        transform_jac += csr_matrix(Jj).dot(M)

        scale_jac = np.zeros(2*self.K)
        scale_jac[:self.K] = 1 / np.sqrt(1 + dx)

        return jac, transform_jac, scale_jac

    def initial_solution(self, sol):
        '''
        Set the initial solution to :class:`BVPSolution` *sol*.
        '''
        self.bvpsol = sol

    def initial_guess(self, fun_list, transform=None, initial_interval=None):
        '''
        Determine the initial polynomial coefficients from guesses for the variables given as a list of functions and, optionally, an initial coordinate transform and initial solution interval.
        '''
        if transform is None:
            print('Calculating initial transform...')
            if initial_interval is None:
                initial_interval = [0, np.copy(self.scale)]
            self.bvpsol.transform.interpolate([lambda t: initial_interval[0]*(1-t) + initial_interval[1]*t, lambda x: x])
            f = lambda coeffs: self.initial_transform(coeffs, fun_list, initial_interval)
            jac = lambda coeffs: self.initial_transform_derivative(coeffs, fun_list)
            _,_ = fsolve(f, np.concatenate([self.bvpsol.transform.get_coeffs(), np.array([self.bvpsol.scale])]), jac=jac, method='nleqres', tol=1e-12, maxiter=100, disp=False)
        else:
            self.bvpsol.transform.interpolate(transform.components)
            self.bvpsol.solution.interpolate(fun_list)

    def initial_transform(self, transform_coeffs, fun_list, initial_interval):
        '''
        Evaluates the residual of the coordinate transform. Used when calculating an initial coordinate transform.
        '''
        self.bvpsol.transform.update_coeffs(transform_coeffs[:-1])
        self.bvpsol.scale = transform_coeffs[-1]
        for n in range(self.dae.N):
            self.bvpsol.solution.components[n].interpolate(lambda t: fun_list[n](self.bvpsol.transform.components[0](t) / self.bvpsol.transform.components[0](1)))

        tres0 = self.monitor(self.collocation_points) - self.bvpsol.transform.components[0].derivative(self.collocation_points)
        tres1 = self.bvpsol.transform.components[0](self.bvpsol.transform.components[1](self.collocation_points)) / self.bvpsol.transform.components[0](1) - self.collocation_points
        cc = self.bvpsol.transform.continuity_error()
        bv = np.array([self.bvpsol.transform.components[0](0) - initial_interval[0], self.bvpsol.transform.components[0](1) - initial_interval[1]])

        return np.concatenate([tres0, tres1, cc, bv])

    def initial_transform_derivative(self, transform_coeffs, fun_list):
        '''
        Calculates the jacobian of the residual of the coordinate transform. Used when calculating an initial coordinate transform.
        '''
        self.bvpsol.transform.update_coeffs(transform_coeffs[:-1])
        self.bvpsol.scale = transform_coeffs[-1]
        y_prime = self.bvpsol.solution.derivative(self.collocation_points)
        derivative = lambda f, x: (f(x+1e-8) - f(x)) / 1e-8
        for n in range(self.dae.N):
            self.bvpsol.solution.components[n].interpolate(lambda t: fun_list[n](self.bvpsol.transform.components[0](t) / self.bvpsol.transform.components[0](1)))
        dx = np.sum(self.bvpsol.solution.derivative(self.collocation_points)[self.dae.dindex]**2, axis=0)

        A = csc_matrix(self.bvpsol.transform.eval_matrix(self.collocation_points))
        B = csc_matrix(self.bvpsol.solution.fitting_matrix())

        Jj = np.zeros((2*self.K,self.dae.N*self.K))

        for n in self.dae.dindex:
            Jj[:self.K,n*self.K:(n+1)*self.K] = np.diag(-self.bvpsol.scale*y_prime[n] / np.sqrt(1 + dx)**3)
        M = csc_matrix(self.bvpsol.solution.derivative_matrix())
        fun_derivative = np.zeros((self.dae.N*self.K,2*self.K))
        for n in range(self.dae.N):
            fun_derivative[n*self.K:(n+1)*self.K,:self.K] = np.diag(derivative(fun_list[n], self.bvpsol.transform.components[0](self.collocation_points) / self.bvpsol.transform.components[0](1)))  / self.bvpsol.transform.components[0](1)
        fun_derivative = csc_matrix(fun_derivative)
        J = csr_matrix(Jj).dot(M).dot(B).dot(fun_derivative).dot(A)

        fun_derivative = np.zeros((self.dae.N*self.K,2))
        for n in range(self.dae.N):
            fun_derivative[n*self.K:(n+1)*self.K,0] = -derivative(fun_list[n], self.bvpsol.transform.components[0](self.collocation_points) / self.bvpsol.transform.components[0](1)) * self.bvpsol.transform.components[0](self.collocation_points) / self.bvpsol.transform.components[0](1)**2
        fun_derivative = csc_matrix(fun_derivative)
        E = csr_matrix(self.bvpsol.transform.eval_matrix(1.0))
        J += csr_matrix(Jj).dot(M).dot(B).dot(fun_derivative).dot(E)

        Jj = np.zeros((2*self.K, 2*self.K))
        Jj[:self.K,:self.K] = -np.eye(self.K)

        M = csc_matrix(self.bvpsol.transform.derivative_matrix())
        J += csr_matrix(Jj).dot(M)

        sigma = self.bvpsol.transform.components[1](self.collocation_points)

        Jj = np.zeros((2*self.K, 2*self.K))
        Jj[self.K:,:self.K] = np.eye(self.K) / self.bvpsol.transform.components[0](1)
        derivative_wrt_sigma = np.concatenate([np.diag(self.bvpsol.transform.components[n].derivative(sigma)) for n in range(2)], axis=0)
        sigma_prime = np.zeros((self.K, self.bvpsol.transform.dimension))
        sigma_prime[:,-self.bvpsol.transform.components[1].dimension:] = self.bvpsol.transform.components[1].eval_matrix(self.collocation_points)
        M = csr_matrix(derivative_wrt_sigma).dot(csc_matrix(sigma_prime))
        M = M + csr_matrix(self.bvpsol.transform.eval_matrix(sigma))
        J += csr_matrix(Jj).dot(M)

        Jj = np.zeros((2*self.K, 2))
        Jj[self.K:,0] = -self.bvpsol.transform.components[0](self.bvpsol.transform.components[1](self.collocation_points)) / self.bvpsol.transform.components[0](1)**2
        M = csr_matrix(self.bvpsol.transform.eval_matrix(1.0))
        J += csr_matrix(Jj).dot(M)

        scale_jac = np.zeros(2*self.K)
        scale_jac[:self.K] = 1 / np.sqrt(1 + dx)

        B0 = self.bvpsol.transform.eval_matrix(0.0)
        B1 = self.bvpsol.transform.eval_matrix(1.0)
        J0 = np.array([[1.0,0], [0,0]])
        J1 = np.array([[0,0], [1.0,0]])

        bv_jac = np.matmul(J0, B0) + np.matmul(J1, B1)

        return bmat([[J, scale_jac[:,None]], [self.bvpsol.transform.continuity_jacobian(), None], [bv_jac, None]], format='csc')

class BVPSolution():
    '''
    Solution to a BVP. This class collects the collocation solution, coordinate transform and coordinate scaling together. The collocation solution (parametrised by the internal coordinate) can be accessed using the :attr:`solution` attribute and the components of the coordinate transform can be accessed using the :attr:`forward` and :attr:`backward` shortcut attributes. Indexing a :class:`BVPSolution` object returns a new :class:`BVPSolution` object with 1-dimensional collocation solution corresponding to the given index and the same coordinate transform.

    The solution can be evaluated at internal coordinates by using the :meth:`evaluate_internal` method or by calling the object like a function. The solution can be evaluated at transformed coordinates by using the :meth:`eval` method.
    '''
    def __init__(self, N, degree, intervals, continuous):
        self.N = N
        self.degree = degree
        self.intervals = intervals
        self.continuous = continuous
        self.solution = CollocationSolution(self.N, degree, np.linspace(0,1,intervals+1), continuous)
        self.scale = 1.0
        self.transform = CollocationSolution(2, degree, np.linspace(0,1,intervals+1), continuous=[True, False])
        self.collocation_points = self.solution.collocation_points
        self.dimension = self.solution.dimension + self.transform.dimension
        self.forward = self.transform[0]
        self.backward = self.transform[1]

    def __call__(self, s):
        return self.evaluate_internal(s)

    def __getitem__(self, n):
        component = BVPSolution(1, self.degree, self.intervals, self.continuous[n])
        component.update_coeffs(np.concatenate([self.solution[n].coeffs, self.transform.get_coeffs(), np.array([self.scale])]))
        return component

    def state(self):
        '''
        Returns an array containing the current coefficients of the collocation solution, coordinate transform and the coordinate scaling.
        '''
        return np.concatenate([self.solution.get_coeffs(), self.transform.get_coeffs(), np.array([self.scale])])

    def update_coeffs(self, coeffs):
        '''
        Update the polynomial coefficients of the collocation solution, coordinate transform and the coordinate scaling.
        '''
        self.solution.update_coeffs(coeffs[:-1-self.transform.dimension])
        self.transform.update_coeffs(coeffs[-1-self.transform.dimension:-1])
        self.scale = coeffs[-1]

    def evaluate_internal(self, t):
        '''
        Evaluate the collocation solution at internal coordinate points *t*. Equivalent to calling the object like a function.
        '''
        return self.solution(t)

    def eval(self, s):
        '''
        Evaluate the collocation solution at transformed points *s*.
        '''
        a = self.forward(0)
        b = self.forward(1)
        x = s-a
        x /= b-a
        return self.solution(self.backward(x))

    def transformed_coordinate(self, t):
        '''
        Returns the coordinate transform evaluated at *t*. Alias of :attr:`forward`.
        '''
        return self.forward(t)

    def scaled_derivative(self, x):
        '''
        Returns the derivative of the collocation solution, scaled by the coordinate transform.
        '''
        return self.solution.derivative(x) / self.forward.derivative(x)

    def scaled_delay(self, x, delay_index):
        '''
        Returns the value in [0,1] that corresponds to the delayed value. The delay must be one of the variables of the system with index *delay_index*.
        '''
        return self.backward(self[delay_index](x))

    def scaled_antiderivative(self, x):
        '''
        Return the integral of the collocation solution with respect to the transformed coordinate.
        '''
        result = np.zeros((self.N, x.shape[0]))
        for n in range(self.N):
            result[n] = (self.forward.deriv()*self.solution[n]).antiderivative(x)
        if self.N == 1:
            result = np.reshape(result, -1)
        return result

    def derivative_wrt_current(self, x, jac=None, nonautonomous_jac=None, sparse=True):
        '''
        Calculate the derivative of the nonlinear system with respect to non-delayed arguments where *jac(x,y)* is a function that returns the jacobian of the system with respect to non-delayed arguments at a point *x*. If the system is non-autonomous then a function *nonautonomous_jac(x,y)* which returns the jacobian of the non-autonmous part with respect to the coordinate transform may be provided and this method will return the corresponding derivative of the nonlinear system. If either *jac* or *nonautonomous_jac* is not provided then the corresponding derivative is zero. The results are returned as :class:`scipy.sparse.csr_matrix` unless *sparse* is `False`, in which case the result is returned as a full numpy array.
        '''
        try:
            K = x.shape[0]
        except AttributeError:
            x = np.array([x])
            K = 1

        if jac is None:
            J = 0
        else:
            try:
                dim = jac(x[0], self).shape[0]
            except AttributeError:
                dim = 1
            Jj = np.zeros((dim*K,self.N*K))
            for k in range(K):
                Jj[k:dim*K:K, k:self.N*K:K] = jac(x[k], self)
            M = csc_matrix(self.solution.eval_matrix(x))
            J = csr_matrix(Jj).dot(M)

        if nonautonomous_jac is None:
            T = 0
        else:
            try:
                dim = nonautonomous_jac(x[0], self).shape[0]
            except AttributeError:
                dim = 1
            Jj = np.zeros((dim*K,2*K))
            for k in range(K):
                Jj[k:dim*K:K, k] = nonautonomous_jac(x[k], self)
            M = csc_matrix(self.transform.eval_matrix(x))
            T = csr_matrix(Jj).dot(M)

        if not sparse:
            if jac is not None:
                J = J.toarray()
            if nonautonomous_jac is not None:
                T = T.toarray()

        return J, T

    def derivative_wrt_derivative(self, x, jac, sparse=True):
        '''
        Calculate the derivative of the nonlinear system with respect to non-delayed derivatives where *jac(x,y)* is a function that returns the jacobian of the system with respect to non-delayed derivatives at a point *x*. Also calculates the corresponding derivative with respect to the coordinate transform. The results are returned as :class:`scipy.sparse.csr_matrix` unless *sparse* is `False`, in which case the result is returned as a full numpy array.
        '''
        try:
            K = x.shape[0]
        except AttributeError:
            x = np.array([x])
            K = 1

        x_scale = self.forward.derivative(x)
        y_prime = self.solution.derivative(x)

        try:
            dim = jac(x[0], self).shape[0]
        except AttributeError:
            dim = 1
        Jj = np.zeros((dim*K,self.N*K))
        for k in range(K):
            Jj[k:dim*K:K, k:self.N*K:K] = jac(x[k], self) / x_scale[k]
        M = csc_matrix(self.solution.derivative_matrix(x))
        J = csr_matrix(Jj).dot(M)

        Jj = np.zeros((dim*K,2*K))
        for k in range(K):
            scaling = -y_prime[:,k] / x_scale[k]**2
            Jj[k:dim*K:K, k] = np.sum(jac(x[k], self) * scaling[None,:], axis=1)
        M = csc_matrix(self.transform.derivative_matrix(x))
        T = csr_matrix(Jj).dot(M)

        if not sparse:
            J = J.toarray()
            T = T.toarray()

        return J, T

    def derivative_wrt_delayed(self, x, jac, delay_index, sparse=True):
        '''
        Calculate the derivative of the nonlinear system with respect to delayed arguments where *jac(x,y)* is a function that returns the jacobian of the system with respect to delayed arguments at a point *x*. The delay must be one of the variables of the system with index *delay_index*. Also calculates the corresponding derivative with respect to the coordinate transform. The results are returned as :class:`scipy.sparse.csr_matrix` unless *sparse* is `False`, in which case the result is returned as a full numpy array. If your system has multiple delays, this function must be used for each delay separately.
        '''
        try:
            K = x.shape[0]
        except AttributeError:
            x = np.array([x])
            K = 1

        x_scale = self.forward.derivative(x)
        if delay_index > 0:
            start = np.sum([c.dimension for c in self[:delay_index]])
        else:
            start = 0
        end = start + self[delay_index].dimension
        sigma_x = self[delay_index](x)
        sigma_t = self.backward(sigma_x)

        try:
            dim = jac(x[0], self).shape[0]
        except AttributeError:
            dim = 1
        Jj = np.zeros((dim*K,self.N*K))
        for k in range(K):
            Jj[k:dim*K:K, k:self.N*K:K] = jac(x[k], self)
        derivative_wrt_sigma = np.concatenate([np.diag(self.solution[n].derivative(sigma_t)) for n in range(self.N)], axis=0)
        sigma_prime = np.zeros((self.collocation_points.shape[0], self.solution.dimension))
        sigma_prime[:,start:end] = diags((self.backward.derivative(sigma_x)), format='csr').dot(self.solution[delay_index].eval_matrix(x))
        M = csr_matrix(derivative_wrt_sigma).dot(csc_matrix(sigma_prime))
        M = csc_matrix(M) + csc_matrix(self.solution.eval_matrix(sigma_t))
        J = csr_matrix(Jj).dot(M)

        sigma_prime = np.zeros((K, self.transform.dimension))
        sigma_prime[:,self.forward.dimension:] = self.backward.eval_matrix(sigma_x)
        M = csr_matrix(derivative_wrt_sigma).dot(csc_matrix(sigma_prime))
        T = csr_matrix(Jj).dot(M)

        if not sparse:
            J = J.toarray()
            T = T.toarray()

        return J, T

    def save(self, savename):
        '''
        Save the solution in compressed format. The file extension `.npz` is appended automatically. The solution can be loaded again using :func:`load_solution`.
        '''
        np.savez_compressed(savename, N=self.N, degree=self.degree, intervals=self.intervals, continuous=self.continuous, state=self.state())

def load_solution(filename):
    '''
    Load a solution from compressed format.
    '''
    data = np.load(filename)
    bvpsol = BVPSolution(data['N'], data['degree'], data['intervals'], data['continuous'])
    bvpsol.update_coeffs(data['state'])
    return bvpsol
