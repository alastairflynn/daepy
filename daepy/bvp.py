import numpy as np
from scipy.sparse import bmat, csc_matrix, csr_matrix
from .collocation import CollocationSolution
from .nonlinear import fsolve

class BVP():
    '''
    This class is used to construct and solve the nonlinear system. It is initialised with a class *dae* which follows the :class:`.DAETemplate`, the *degree* of the differential variables and the number of *intervals* in the mesh.
    '''
    def __init__(self, dae, degree=3, intervals=10):
        self.dae = dae
        continuous = numpy.array([False for n in range(dae.N)])
        continuous[dae.dindex] = True
        self.bvpsol = BVPSolution(degree, intervals, continuous)

        self.collocation_points = self.bvpsol.collocation_points
        self.K = self.collocation_points.shape[0]
        self.breakpoints = np.copy(self.bvpsol.breakpoints)
        self.dimension = self.bvpsol.dimension

    def solve(self, method='nleqres', tol=1e-8, maxiter=100, disp=False):
        '''
        Solve the nonlinear system where *method* is one of

        * `'nleqres'` a damped global Newton method [1]_ (the default)
        * `'lm'` the Leveberg--Marquardt method [2]_
        * `'partial_inverse'` Newton-like method which calculates a partial inverse of the Jacobian by calculating a QR decomposition and doing a partial backwards substitution when the step doesn’t converge
        * `'steepest_descent'` steepest descent method

        *tol* is required residual tolerance, *maxiter* is the maximum number of iterations and *disp* controls whether convergence messages are printed.

        .. [1] P. Deuflhard. Systems of Equations: Global Newton Methods. In *Newton Methods for Nonlinear Problems*, Springer Series in Computational Mathematics, pages 109–172. Springer, Berlin, Heidelberg, 2011.
        .. [2] J. Dennis and R. Schnabel. *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*. Classics in Applied Mathematics. Society for Industrial and Applied Mathematics, January 1996.
        '''
        c,m = fsolve(self.eval, self.state(), self.jac, method, tol, maxiter, disp)
        self.bvpsol.update_coeffs(c)
        return m

    def monitor(self, x):
        '''
        The monitor function used to define the coordinate transform.
        '''
        dx = np.sum(self.bvpsol.solution.derivative(x)[self.dae.dindex]**2, axis=0)
        return self.bvpsol.scale / np.sqrt(1 + dx)

    def state(self):
        '''
        Returns an array containing the current coefficients of the collocation solution, coordinate transform and the coordinate scaling.
        '''
        return self.solution.state()

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

    def initial_guess(self, fun_list, transform=None, initial_interval=None):
        '''
        Determine the initial polynomial coefficients from initial guesses for the variables given as a list of functions and, optionally, an initial coordinate transform and initial solution interval.
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
        y_prime = self.solution.derivative(self.collocation_points)
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

        return bmat([[J, scale_jac[:,None]], [self.transform.continuity_jacobian(), None], [bv_jac, None]], format='csc')

class BVPSolution():
    '''
    Solution to a BVP. This class collects the collocation solution, coordinate transform and coordinate scaling together.
    '''
    def __init__(self, degree, intervals, continuous):
        self.degree = degree
        self.intervals = intervals
        self.continuous = continuous
        self.solution = CollocationSolution(N, degree, np.linspace(0,1,intervals+1), continuous)
        self.scale = 1.0
        self.transform = CollocationSolution(2, degree, np.linspace(0,1,intervals+1), continuous=[True, False])
        self.collocation_points = self.solution.collocation_points
        self.dimension = self.solution.dimension + self.transform.dimension

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

    def scaled_derivative(self, x):
        '''
        Returns the derivative of the collocation solution, scaled by the coordinate transform.
        '''
        return self.solution.derivative(x) / transform.components[0].derivative(x)

    def save(self, savename):
        '''
        Save the solution in compressed format. The file extension `.npz` is appended automatically. The solution can be loaded again using :func:`load_solution`.
        '''
        np.savez_compressed(savename, degree=self.degree, intervals=self.intervals, continuous=self.continuous, state=self.state())

def load_solution(filename):
    '''
    Load a solution from compressed format.
    '''
    data = np.load(filename)
    bvpsol = BVPSolution(data['degree'], data['intervals'], data['continuous'])
    bvpsol.update_coeffs(data['state'])
    return bvpsol
