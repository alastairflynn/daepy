import numpy as np
from scipy.linalg import norm, det
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from .nonlinear import fsolve

class BVPContinuation():
    '''
    Used to perform continuation runs. It is initialised with a :class:`.BVP` *bvp* and *method* is one of

    * `'pseudo_arclength'` the pseudo-arclength method [1]_ (the default)
    * `'naive'` naive continuation with no predictive step

    Once initialised, a continuation run may be performed using the :meth:`continuation_run` method.

    .. note::
        The *bvp* must have been initialised with a *dae* object that defines the :meth:`.update_parameter` method and all the jacobian methods. In the future, continuation using a finite difference approximation of the jacobian may be supported although it would still be strongly recommended to use an analytic jacobian, if available.

    .. [1] E. Allgower and K. Georg. *Introduction to Numerical Continuation Methods*. Classics in Applied Mathematics. Society for Industrial and Applied Mathematics, January 2003.
    '''
    def __init__(self, bvp, method='pseudo_arclength'):
        self.method = method
        self.bvp = bvp
        self.tangent = None
        self.arclength = 0.0
        self.y0 = None

    def eval(self, y):
        '''
        Evaluate the augmented system.
        '''
        self.bvp.dae.update_parameter(y[-1])
        step = np.dot(self.tangent, y - self.y0) - self.arclength
        return np.concatenate([self.bvp.eval(y[:-1]), np.array([step])])

    def jac(self, y):
        '''
        Evaluate the augmented jacobian.
        '''
        self.bvp.dae.update_parameter(y[-1])
        jac = self.bvp.jac(y[:-1])
        param_jac = self.bvp.param_jac(self.bvp.collocation_points, self.bvp.bvpsol)
        return bmat([[jac, param_jac[:,None]], [self.tangent[None,:-1], self.tangent[-1]]], format='csc')

    def find_tangent(self, y):
        '''
        Find the tangent of the solution path.
        '''
        self.bvp.dae.update_parameter(y[-1])
        jac = self.bvp.jac(y[:-1])
        param_jac = self.bvp.param_jac(self.bvp.collocation_points, self.bvp.bvpsol)
        determinant = det(jac.toarray())
        jac = csr_matrix(jac)
        z = spsolve(jac, -param_jac, use_umfpack=True)
        self.tangent = np.concatenate([z, np.array([1.0])]) / np.sqrt(1 + np.dot(z,z))
        return determinant

    def continuation_step(self, y0, d, stepsize=1.0, adaptive=True, target=None, tol=1e-8, maxiter=100):
        '''
        Perform a sing continuation step.
        '''
        if self.method == 'pseudo_arclength':
            self.y0 = np.copy(y0)
            determinant = self.find_tangent(self.y0)
            # if np.sign(d*determinant) < 0:
            #     print('Located turning point')
            #     stepsize *= -1.0

            self.arclength = stepsize / self.tangent[-1]
            y1 = y0 + self.arclength*self.tangent
            self.bvp.dae.update_parameter(y1[-1])

            if adaptive:
                old_res = norm(self.eval(y0))
                res = norm(self.eval(y1))
                while res >= 0.25*old_res and stepsize > np.sqrt(tol):
                    stepsize /= 2
                    self.arclength = stepsize #/ self.tangent[-1]
                    y1 = y0 + self.arclength*self.tangent
                    self.bvp.dae.update_parameter(y1[-1])
                    old_res = norm(self.eval(y0))
                    res = norm(self.eval(y1))

            try:
                y, m = fsolve(self.eval, y1, jac=self.jac, method='nleqres', tol=tol, maxiter=maxiter, disp=False)
                result = [y, m, y[-1] - y0[-1], determinant]
            except:
                result = [y0, 0, 0.0, determinant]
        else:
            p = y0[-1] + stepsize
            self.bvp.dae.update_parameter(p)

            x, m = fsolve(self.bvp.eval, y0[:-1], jac=self.bvp.jac, method='nleqres', tol=tol, maxiter=maxiter, disp=False)
            y = np.concatenate([x, np.array([p])])
            result = [y, m, stepsize, d]
        return result

    def continuation_run(self, x0, p0, steps=1, stepsize=1.0, target=None, tol=1e-8, maxiter=100, disp=False, callback=None):
        '''
        Perform a continuation run where *x0* is the initial guess (typically one would use :code:`bvp.state()`), *p0* is the initial value of the parameter, *steps* is either a maximum number of steps or a numpy array of parameter values which determine the steps explicitly, *stepsize* is the initial stepsize (the pseudo-arclength method will adapt the stepsize, ignored if steps are given explicitly), *target* is a value for the parameter at which the continuation will stop (optional), *tol* is the required solution tolerance, *maxiter* is the maximum number of iterations for the nonlinear solver, *disp* determines whether to print progress messages and *callback(parameter, solution)* is a function that will be called before each continuation step, for example to draw a new line on a plot at each step (optional).

        The function returns the final solution and final parameter value. The `bvp` object is updated during the continuation run so :code:`bvp.state()` will correspond to the final solution and the parameter value in `bvp` will correspond to the final parameter value as well.

        .. note::
            When using the :code:`pseudo_arclength` method, setting a *target* or expilicity giving *steps* does not guarantee that the parameter value of the solution will correspond to the given value. If you wish to use parameter continuation to reach a specific parameter value, specify *target* or give explicit *steps* to get close to the desired parameter value and then use :meth:`.solve` with the exact parameter value.
        '''
        x, m = fsolve(self.bvp.eval, x0, jac=self.bvp.jac, method='nleqres', tol=tol, maxiter=maxiter, disp=False)
        y = np.concatenate([x, np.array([p0])])
        residual = self.bvp.eval(x)
        if disp:
            print('{:<5} {:<15} {:<15} {:<15} {:<}'.format('Step', 'Parameter', 'Residual', 'Iterations', 'Stepsize'))
            print('{:<5d} {:<15e} {:<15e} {:<15d}'.format(0, y[-1], norm(residual, ord=2), m))
            # print('00', y[-1], norm(residual, ord=2), m)

        if callback is not None:
            callback(y[-1], self.bvp.bvpsol)

        try:
            n_steps = len(steps)
            adaptive = False
        except:
            n_steps = steps
            adaptive = True

        d = 0.0
        for step in range(1,n_steps+1):
            if not adaptive:
                stepsize = steps[step-1] - y[-1]
            y, m, a, d = self.continuation_step(y, d, stepsize=stepsize, adaptive=adaptive, target=target, tol=tol, maxiter=maxiter)
            self.bvp.dae.update_parameter(y[-1])
            res = norm(self.bvp.eval(y[:-1]))
            if disp:
                # print('%02d'%step, y[-1], res, m, a)
                print('{:<5d} {:<15e} {:<15e} {:<15d} {:<e}'.format(step, y[-1], norm(residual, ord=2), m, a))
            # stepsize *= np.sign(a*stepsize)
            # stepsize = a

            if callback is not None:
                callback(y[-1], self.bvp.bvpsol)

            if target is not None and np.sign(stepsize)*y[-1] >= np.sign(stepsize)*target:
                print('Reached target')
                break

            if res > np.sqrt(tol) or m == 0:
                print('Convergence failed')
                y = np.copy(self.y0)
                r = self.eval(y)
                break

        return y[:-1], y[-1]
