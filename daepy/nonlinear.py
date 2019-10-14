import numpy as np
from scipy.linalg import norm, solve, qr, solve_triangular
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve
import dill
from multiprocessing import Pool
from .derivatives import approx_jac

def fsolve(fun, x0, jac=None, method='nleqres', tol=1e-8, maxiter=100, disp=False):
    '''
    Solve a nonlinear system where *fun* is a function that evaluates the nonlinear system, *x0* is the initial guess, *jac* is a function that evaluates the jacobian of the system, *method* is one of

    * `'nleqres'` a damped global Newton method [1]_ (the default)
    * `'lm'` the Leveberg--Marquardt method [2]_
    * `'partial_inverse'` Newton-like method which calculates a partial inverse of the Jacobian by calculating a QR decomposition and doing a partial backwards substitution when the step doesn’t converge
    * `'steepest_descent'` steepest descent method

    *tol* is required residual tolerance, *maxiter* is the maximum number of iterations and *disp* controls whether convergence messages are printed. If *jac* is `None` then a finite difference approximation of the jacobian is used.

    .. [1] P. Deuflhard. Systems of Equations: Global Newton Methods. In *Newton Methods for Nonlinear Problems*, Springer Series in Computational Mathematics, pages 109–172. Springer, Berlin, Heidelberg, 2011.
    .. [2] J. Dennis and R. Schnabel. *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*. Classics in Applied Mathematics. Society for Industrial and Applied Mathematics, January 1996.
    '''
    if jac is None:
        stream = dill.dumps(fun)
        pool = Pool()

    x = np.copy(x0)
    res = fun(x)
    cost = norm(res, ord=2)
    if disp:
        print('{:<5} {:<15} {:<}'.format('Iter', 'Residual', 'Stepsize'))
        print('{:<5d} {:<15e}'.format(0, cost))
        # print('Initial cost:', cost)

    m = 0
    if method=='lm':
        l = 0.0
    else:
        l = 1.0
    old = np.copy(x0)
    old_cost = np.inf
    while cost > tol and m < maxiter and cost < old_cost - tol/10:
        old = np.copy(x)
        old_cost = np.copy(cost)

        if jac is None:
            J = approx_jac(stream, x, 1e-8, pool)
        else:
            J = jac(x)

        if method=='lm':
            x, cost, inner_iterations, l = lm(fun, x, J, l, maxiter=maxiter)
        elif method=='nleqres':
            x, cost, inner_iterations, l = nleqres(fun, x, J, l, maxiter=maxiter)
        elif method=='partial_inverse':
            x, cost, inner_iterations, l = partial_inverse(fun, x, J)
        else:
            x, cost, inner_iterations, l = steepest_descent(fun, x, J, l, maxiter=maxiter)

        m += inner_iterations
        if disp:
            # print(m, cost, l)
            print('{:<5d} {:<15e} {:<e}'.format(m, cost, l))

        if method=='lm':
            l /= 4
            if l < 1e-14:
                l = 0.0
        elif method=='nleqres':
            # l = min(4*l, 1.0)
            pass
        else:
            l = min(4*l, 1.0)

    if cost > old_cost:
        x = np.copy(old)
        cost = norm(fun(x), ord=2)

    if disp:
        print('Final cost:', cost)

    if jac is None:
        pool.close()
    return x, m

def partial_inverse(fun, x, J):
    res = fun(x)
    cost = norm(res, ord=2)

    old = np.copy(x)
    old_res = np.copy(res)
    old_cost = np.copy(cost)

    J = J.toarray()

    Q, R, P = qr(J, overwrite_a=True, mode='full', pivoting=True, check_finite=False)
    mag = np.abs(np.diag(R))
    order = np.argsort(mag)
    a = 0 #np.count_nonzero(np.less(mag, 1e-8*np.max(mag)))
    Qf = Q.T.dot(res)
    RQf = np.zeros(res.shape)
    delta_x = np.zeros(J.shape[1])

    mask = np.greater_equal(order, a)
    try:
        RQf[mask] = solve_triangular(R[mask][:,mask], Qf[mask], overwrite_b=False)
        RQf[~mask] = 0.0
        delta_x[P] = RQf

        x = old - delta_x
        res = fun(x)
        cost = norm(res)
    except:
        cost = np.inf

    while cost > old_cost and a < J.shape[1]-1:
        a += 1
        mask = np.greater_equal(order, a)
        try:
            RQf[mask] = solve_triangular(R[mask][:,mask], Qf[mask], overwrite_b=False)
            RQf[~mask] = 0.0
            delta_x[P] = RQf

            x = old - delta_x
            res = fun(x)
            cost = norm(res)
        except:
            cost = np.inf

    return x, cost, 1, a

def nleqres(fun, x, J, a=1, maxiter=100):
    res = fun(x)
    cost = norm(res, ord=2)

    old = np.copy(x)
    old_res = np.copy(res)
    old_cost = np.copy(cost)

    J = csc_matrix(J)
    delta_x = spsolve(J, res, use_umfpack=True)

    if np.any(np.isnan(delta_x)):
        print('Singular Jacobian, performing partial solve...')
        J = J.toarray()
        Q, R, P = qr(J, overwrite_a=True, mode='full', pivoting=True, check_finite=False)
        mag = np.abs(np.diag(R))
        Qf = Q.T.dot(res)
        RQf = np.zeros(res.shape)
        delta_x = np.zeros(J.shape[1])

        mask = ~np.isclose(mag, 0) #np.greater_equal(mag, 1e-6*np.max(mag))
        RQf[mask] = solve_triangular(R[mask][:,mask], Qf[mask], overwrite_b=False)
        RQf[~mask] = 0.0
        delta_x[P] = RQf

    a_old = 0.0
    while a > 1e-14:
        x = old - a*delta_x
        res = fun(x)
        if np.any(np.isnan(res)):
            a /= 2
            continue
        cost = norm(res)

        theta = cost / old_cost
        mu_prime = old_cost*a**2 / (2*norm(res - (1-a)*old_res))

        if theta >= 1:
            # a /= 2
            a = min(mu_prime, a/2)
        else:
            a_prime = min(mu_prime, 1)
            if a_prime >= 4*a and a > a_old:
                a_old = a
                a = a_prime
                # rejected = True
            # else:
            break

    return x, cost, 1, a

def lm(fun, x, J, l=0, maxiter=100):
    old = np.copy(x)
    res = fun(x)
    init_cost = norm(res, ord=2)

    J = csc_matrix(J)
    JJ = J.T.dot(J)
    damping = diags(np.ones(J.shape[1]), format='csr') #JJ.diagonal() np.ones(J.shape[1])
    Jf = J.T.dot(res)

    # l = 0 #old_cost
    search = spsolve(JJ + l*damping, Jf, use_umfpack=True)
    x = old - search
    res = fun(x)
    cost = norm(res, ord=2)
    while cost > init_cost and l < 1e8:
        if l < 1e-16:
            l = 1e-8
        else:
            l *= 2
        search = spsolve(JJ + l*damping, Jf, use_umfpack=True)
        x = old - search
        res = fun(x)
        cost = norm(res, ord=2)

    return x, cost, 1, l

def steepest_descent(fun, x, J, a=1, maxiter=100):
    inner_iterations = 0
    res = fun(x)
    cost = norm(res, ord=2)

    J = csc_matrix(J)

    while inner_iterations == 0 or cost < old_cost and inner_iterations < 1:
        old = np.copy(x)
        old_cost = np.copy(cost)
        a = min(a*2, 1.0)

        search = J.T.dot(res)
        x = old - a*search
        res = fun(x)
        cost = norm(res, ord=2)
        while cost > old_cost and a > 1e-14:
            a *= 0.5
            x = old - a*search
            res = fun(x)
            cost = norm(res, ord=2)

        inner_iterations += 1

    return x, cost, 1, a
