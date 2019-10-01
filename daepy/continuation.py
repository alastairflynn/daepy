import numpy as np
from scipy.linalg import norm, det
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import spsolve
from nonlinear import fsolve

class BVPContinuation():
    def __init__(self, bvp, update_parameter, param_jac, method='pseudo_arclength'):
        self.method = method
        self.bvp = bvp
        self.update_parameter = update_parameter
        self.param_jac = param_jac
        self.tangent = None
        self.arclength = 0.0
        self.y0 = None

    def eval(self, y):
        self.update_parameter(self.bvp, y[-1])
        step = np.dot(self.tangent, y - self.y0) - self.arclength
        return np.concatenate([self.bvp.eval(y[:-1]), np.array([step])])

    def jac(self, y):
        self.update_parameter(self.bvp, y[-1])
        jac = self.bvp.jac(y[:-1])
        param_jac = self.param_jac(y[:-1], y[-1])
        return bmat([[jac, param_jac[:,None]], [self.tangent[None,:-1], self.tangent[-1]]], format='csc')

    def find_tangent(self, y):
        self.update_parameter(self.bvp, y[-1])
        jac = self.bvp.jac(y[:-1])
        param_jac = self.param_jac(y[:-1], y[-1])
        determinant = det(jac.toarray())
        jac = csr_matrix(jac)
        z = spsolve(jac, -param_jac)
        self.tangent = np.concatenate([z, np.array([1.0])]) / np.sqrt(1 + np.dot(z,z))
        return determinant

    def continuation_step(self, y0, d, stepsize=1.0, adaptive=True, target=None, tol=1e-8, maxiter=100):
        if self.method == 'pseudo_arclength':
            self.y0 = np.copy(y0)
            determinant = self.find_tangent(self.y0)
            # if np.sign(d*determinant) < 0:
            #     print('Located turning point')
            #     stepsize *= -1.0

            self.arclength = stepsize / self.tangent[-1]
            y1 = y0 + self.arclength*self.tangent
            self.update_parameter(self.bvp, y1[-1])

            if adaptive:
                old_res = norm(self.eval(y0))
                res = norm(self.eval(y1))
                while res >= 0.25*old_res and stepsize > np.sqrt(tol):
                    stepsize /= 2
                    self.arclength = stepsize #/ self.tangent[-1]
                    y1 = y0 + self.arclength*self.tangent
                    self.update_parameter(self.bvp, y1[-1])
                    old_res = norm(self.eval(y0))
                    res = norm(self.eval(y1))

            try:
                y, m = fsolve(self.eval, y1, jac=self.jac, method='nleqres', tol=tol, maxiter=maxiter, disp=False)
                result = [y, m, y[-1] - y0[-1], determinant]
            except:
                result = [y0, 0, 0.0, determinant]
        else:
            p = y0[-1] + stepsize
            self.update_parameter(self.bvp, p)

            x, m = fsolve(self.bvp.eval, y0[:-1], jac=self.bvp.jac, method='nleqres', tol=tol, maxiter=maxiter, disp=False)
            y = np.concatenate([x, np.array([p])])
            result = [y, m, stepsize, d]
        return result

    def continuation_run(self, x0, p0, steps=1, stepsize=1.0, target=None, tol=1e-8, maxiter=100, disp=False, callback=None):
        x, m = fsolve(self.bvp.eval, x0, jac=self.bvp.jac, method='nleqres', tol=tol, maxiter=maxiter, disp=False)
        y = np.concatenate([x, np.array([p0])])
        residual = self.bvp.eval(x)
        if disp:
            print('00', y[-1], norm(residual, ord=2), m)

        if callback is not None:
            callback(self.bvp)

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
            self.update_parameter(self.bvp, y[-1])
            res = norm(self.bvp.eval(y[:-1]))
            if disp:
                print('%02d'%step, y[-1], res, m, a)
            # stepsize *= np.sign(a*stepsize)
            # stepsize = a

            if callback is not None:
                callback(self.bvp)

            if np.sign(stepsize)*y[-1] >= np.sign(stepsize)*target:
                print('Reached target')
                break

            if res > np.sqrt(tol) or m == 0:
                print('Convergence failed')
                y = np.copy(self.y0)
                r = self.eval(y)
                break

        return y[:-1], y[-1]
