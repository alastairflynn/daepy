import numpy as np
from collocation import CollocationSolution, UnivariateCollocationSolution
from scipy.sparse import bmat, csc_matrix, csr_matrix
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from nonlinear import fsolve

class BVPSolution():
    '''
    Solution to a BVP. This class collects the collocation solution, coordinate transform and coordinate scaling together.
    '''
    def __init__(self, solution, scale):
        self.solution = solution
        self.scale = scale
        self.transform = CollocationSolution(2, self.solution.degree, self.solution.breakpoints, continuous=[True, False])

class BVP():
    '''
    This class is used to construct and solve the nonlinear system.
    '''
    def __init__(self, dae, solution, scale, transform=None):
        self.dae = dae
        self.solution = solution
        self.scale = scale
        self.transform = CollocationSolution(2, self.solution.degree, self.solution.breakpoints, continuous=[True, False])

        self.K = self.solution.collocation_points.shape[0]
        self.breakpoints = np.copy(self.solution.breakpoints)
        self.discontinuities = None
        self.dimension = self.solution.dimension + self.transform.dimension

    def monitor(self, x):
        dx = np.sum(self.solution.derivative(x)[self.dae.dindex]**2, axis=0)
        return self.scale / np.sqrt(1 + dx)

    def state(self):
        return np.concatenate([self.solution.get_coeffs(), self.transform.get_coeffs(), np.array([self.scale])])

    def eval(self, coeffs):
        self.solution.update_coeffs(coeffs[:-1-self.transform.dimension])
        self.transform.update_coeffs(coeffs[-1-self.transform.dimension:-1])
        self.scale = coeffs[-1]

        residual = self.dae.fun(self.solution.collocation_points, self.solution, self.transform).reshape(-1)
        bv_res = self.dae.bv(self.solution, self.transform).reshape(-1)

        tres0 = self.monitor(self.solution.collocation_points) - self.transform.components[0].derivative(self.solution.collocation_points)
        tres1 = self.transform.components[0](self.transform.components[1](self.solution.collocation_points)) / self.transform.components[0](1) - self.solution.collocation_points

        cc = self.solution.continuity_error()
        cc_transform = self.transform.continuity_error()

        res = np.concatenate([residual, tres0, tres1, cc, cc_transform, bv_res])

        return res

    def jac(self, coeffs):
        self.solution.update_coeffs(coeffs[:-1-self.transform.dimension])
        self.transform.update_coeffs(coeffs[-1-self.transform.dimension:-1])
        self.scale = coeffs[-1]

        jac, transform_jac = self.dae.jacobian(self.solution.collocation_points, self.solution, self.transform)
        bv_jac, bv_transform_jac = self.dae.bv_jacobian(self.solution, self.transform)

        t_jac, t_transform_jac, t_scale_jac = self.monitor_derivative(self.solution.collocation_points)

        cc_jac = self.solution.continuity_jacobian()
        cc_transform_jac = self.transform.continuity_jacobian()

        J = bmat([[jac, transform_jac, None], [t_jac, t_transform_jac, t_scale_jac[:,None]], [cc_jac, None, None], [None, cc_transform_jac, None], [bv_jac, bv_transform_jac, None]], format='csc')

        J.eliminate_zeros()

        return J

    def monitor_derivative(self, x):
        y = self.solution
        y_prime = y.derivative(x)
        dx = np.sum(y_prime[self.dae.dindex]**2, axis=0)

        Jj = np.zeros((2*self.K,self.dae.N*self.K))

        for n in self.dae.dindex:
            Jj[:self.K,n*self.K:(n+1)*self.K] = np.diag(-self.scale*y_prime[n] / np.sqrt(1 + dx)**3)
        M = csc_matrix(y.derivative_matrix())
        jac = csr_matrix(Jj).dot(M)

        Jj = np.zeros((2*self.K, 2*self.K))
        Jj[:self.K,:self.K] = -np.eye(self.K)

        M = csc_matrix(self.transform.derivative_matrix())
        transform_jac = csr_matrix(Jj).dot(M)

        sigma = self.transform.components[1](x)

        Jj = np.zeros((2*self.K, 2*self.K))
        Jj[self.K:,:self.K] = np.eye(self.K) / self.transform.components[0](1)
        derivative_wrt_sigma = np.concatenate([np.diag(self.transform.components[n].derivative(sigma)) for n in range(2)], axis=0)
        sigma_prime = np.zeros((self.K, self.transform.dimension))
        sigma_prime[:,-self.transform.components[1].dimension:] = self.transform.components[1].eval_matrix(x)
        M = csr_matrix(derivative_wrt_sigma).dot(csc_matrix(sigma_prime))
        M = M + csr_matrix(self.transform.eval_matrix(sigma))
        transform_jac += csr_matrix(Jj).dot(M)

        Jj = np.zeros((2*self.K, 2))
        Jj[self.K:,0] = -self.transform.components[0](self.transform.components[1](self.solution.collocation_points)) / self.transform.components[0](1)**2
        M = csr_matrix(self.transform.eval_matrix(1.0))
        transform_jac += csr_matrix(Jj).dot(M)

        scale_jac = np.zeros(2*self.K)
        scale_jac[:self.K] = 1 / np.sqrt(1 + dx)

        return jac, transform_jac, scale_jac

    def initial_guess(self, fun_list, transform=None, initial_interval=None):
        if transform is None:
            print('Calculating initial transform...')
            if initial_interval is None:
                initial_interval = [0, np.copy(self.scale)]
            self.transform.interpolate([lambda t: initial_interval[0]*(1-t) + initial_interval[1]*t, lambda x: x])
            f = lambda coeffs: self.initial_transform(coeffs, fun_list, initial_interval)
            jac = lambda coeffs: self.initial_transform_derivative(coeffs, fun_list)
            _,_ = fsolve(f, np.concatenate([self.transform.get_coeffs(), np.array([self.scale])]), jac=jac, method='nleqres', tol=1e-12, maxiter=100, disp=False)
        else:
            self.transform.interpolate(transform.components)
            self.solution.interpolate(fun_list)

    def initial_transform(self, transform_coeffs, fun_list, initial_interval):
        self.transform.update_coeffs(transform_coeffs[:-1])
        self.scale = transform_coeffs[-1]
        for n in range(self.dae.N):
            self.solution.components[n].interpolate(lambda t: fun_list[n](self.transform.components[0](t) / self.transform.components[0](1)))

        tres0 = self.monitor(self.solution.collocation_points) - self.transform.components[0].derivative(self.solution.collocation_points)
        tres1 = self.transform.components[0](self.transform.components[1](self.solution.collocation_points)) / self.transform.components[0](1) - self.solution.collocation_points
        cc = self.transform.continuity_error()
        bv = np.array([self.transform.components[0](0) - initial_interval[0], self.transform.components[0](1) - initial_interval[1]])

        return np.concatenate([tres0, tres1, cc, bv])

    def initial_transform_derivative(self, transform_coeffs, fun_list):
        self.transform.update_coeffs(transform_coeffs[:-1])
        self.scale = transform_coeffs[-1]
        y_prime = self.solution.derivative(self.solution.collocation_points)
        derivative = lambda f, x: (f(x+1e-8) - f(x)) / 1e-8
        for n in range(self.dae.N):
            self.solution.components[n].interpolate(lambda t: fun_list[n](self.transform.components[0](t) / self.transform.components[0](1)))
        dx = np.sum(self.solution.derivative(self.solution.collocation_points)[self.dae.dindex]**2, axis=0)

        A = csc_matrix(self.transform.eval_matrix(self.solution.collocation_points))
        B = csc_matrix(self.solution.fitting_matrix())

        Jj = np.zeros((2*self.K,self.dae.N*self.K))

        for n in self.dae.dindex:
            Jj[:self.K,n*self.K:(n+1)*self.K] = np.diag(-self.scale*y_prime[n] / np.sqrt(1 + dx)**3)
        M = csc_matrix(self.solution.derivative_matrix())
        fun_derivative = np.zeros((self.dae.N*self.K,2*self.K))
        for n in range(self.dae.N):
            fun_derivative[n*self.K:(n+1)*self.K,:self.K] = np.diag(derivative(fun_list[n], self.transform.components[0](self.solution.collocation_points) / self.transform.components[0](1)))  / self.transform.components[0](1)
        fun_derivative = csc_matrix(fun_derivative)
        J = csr_matrix(Jj).dot(M).dot(B).dot(fun_derivative).dot(A)

        fun_derivative = np.zeros((self.dae.N*self.K,2))
        for n in range(self.dae.N):
            fun_derivative[n*self.K:(n+1)*self.K,0] = -derivative(fun_list[n], self.transform.components[0](self.solution.collocation_points) / self.transform.components[0](1)) * self.transform.components[0](self.solution.collocation_points) / self.transform.components[0](1)**2
        fun_derivative = csc_matrix(fun_derivative)
        E = csr_matrix(self.transform.eval_matrix(1.0))
        J += csr_matrix(Jj).dot(M).dot(B).dot(fun_derivative).dot(E)

        Jj = np.zeros((2*self.K, 2*self.K))
        Jj[:self.K,:self.K] = -np.eye(self.K)

        M = csc_matrix(self.transform.derivative_matrix())
        J += csr_matrix(Jj).dot(M)

        sigma = self.transform.components[1](self.solution.collocation_points)

        Jj = np.zeros((2*self.K, 2*self.K))
        Jj[self.K:,:self.K] = np.eye(self.K) / self.transform.components[0](1)
        derivative_wrt_sigma = np.concatenate([np.diag(self.transform.components[n].derivative(sigma)) for n in range(2)], axis=0)
        sigma_prime = np.zeros((self.K, self.transform.dimension))
        sigma_prime[:,-self.transform.components[1].dimension:] = self.transform.components[1].eval_matrix(self.solution.collocation_points)
        M = csr_matrix(derivative_wrt_sigma).dot(csc_matrix(sigma_prime))
        M = M + csr_matrix(self.transform.eval_matrix(sigma))
        J += csr_matrix(Jj).dot(M)

        Jj = np.zeros((2*self.K, 2))
        Jj[self.K:,0] = -self.transform.components[0](self.transform.components[1](self.solution.collocation_points)) / self.transform.components[0](1)**2
        M = csr_matrix(self.transform.eval_matrix(1.0))
        J += csr_matrix(Jj).dot(M)

        scale_jac = np.zeros(2*self.K)
        scale_jac[:self.K] = 1 / np.sqrt(1 + dx)

        B0 = self.transform.eval_matrix(0.0)
        B1 = self.transform.eval_matrix(1.0)
        J0 = np.array([[1.0,0], [0,0]])
        J1 = np.array([[0,0], [1.0,0]])

        bv_jac = np.matmul(J0, B0) + np.matmul(J1, B1)

        return bmat([[J, scale_jac[:,None]], [self.transform.continuity_jacobian(), None], [bv_jac, None]], format='csc')
