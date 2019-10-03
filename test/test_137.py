import numpy as np
# from scipy.sparse import csr_matrix, csc_matrix, bmat

class DAE():
    def __init__(self, alpha):
        self.N = 2
        self.dindex = [0]
        self.aindex = [1]
        self.alpha = alpha

    def fun(self, x, y):
        try:
            K = x.shape[0]
        except:
            K = 1
            x = np.array([x])
        yx = y.solution(x)
        y_prime = y.scaled_derivative(x)
        sigma = y.scaled_delay(x, 1)
        s = y.transformed_coordinate(x)

        r = np.zeros((self.N,K))

        r[0] = y.components[0].eval(sigma) + (3+self.alpha)*s**(2+self.alpha) - s**((3+self.alpha)**2) - y_prime[0]
        r[1] = yx[1] - yx[0]

        if K==1:
            r = r[:,0]
        return r

    def bv(self, y):
        r = np.array([y(0)[0] - 0.0, y.forward(0) - 0, y.forward(1) - 1.0])
        return r

    def jacobian(self, x, y):
        jac, transform_jac = y.derivative_wrt_current(x, self.standard_jac, self.nonautonomous_jac)

        J, T = y.derivative_wrt_derivative(x, self.derivative_jac)
        jac += J
        transform_jac += T

        J, T = y.derivative_wrt_delayed(x, self.delayed_jac, 1)
        jac += J
        transform_jac += T

        return jac, transform_jac

    def bv_jacobian(self, y):
        bv_jac, bv_transform_jac = y.derivative_wrt_current(0.0, self.bv_jac0, self.bv_transform_jac0)
        B1, T1 = y.derivative_wrt_current(1.0, None, self.bv_transform_jac1)
        bv_transform_jac += T1

        return bv_jac, bv_transform_jac

    def standard_jac(self, x, y):
        J = np.zeros((self.N, self.N))
        J[1,1] = 1.0
        J[1,0] = -1.0
        return J

    def nonautonomous_jac(self, x, y):
        s = y.transformed_coordinate(x)
        J = np.zeros(self.N)
        J[0] = (3+self.alpha)*(2+self.alpha)*s**(2+self.alpha-1) - ((3+self.alpha)**2)*s**((3+self.alpha)**2 - 1)
        return J

    def derivative_jac(self, x, y):
        diag = np.zeros(self.N)
        diag[0] = -1.0
        return np.diag(diag)

    def delayed_jac(self, x, y):
        J = np.zeros((self.N, self.N))
        J[0,0] = 1.0
        return J

    def bv_jac0(self, x, y):
        r = np.zeros((3,self.N))
        r[0,0] = 1.0
        return r

    def bv_transform_jac0(self, x, y):
        r = np.zeros(3)
        r[1] = 1.0
        return r

    def bv_transform_jac1(self, x, y):
        r = np.zeros(3)
        r[2] = 1.0
        return r
