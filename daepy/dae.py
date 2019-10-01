class DAE():
    '''
    The user must write a class with this template to define the BVP. The class must define the attributes N, dindex and aindex and the methods fun, bv and, optionally, jacobian and bv_jacobian. You can of course add your own attributes and methods.
    '''
    def __init__(self):
        self.N = 2
        self.dindex = [0]
        self.aindex = [1]

    def fun(self, x, y, transform):
        '''
        Evaluate the system.
        '''
        # write your system here
        residual = None
        return residual

    def bv(self, y, transform):
        '''
        Evaluate the boundary conditions.
        '''
        # write your boundary conditions here
        bc = None
        return bc

    def jacobian(self, x, y, transform):
        '''
        Calculate the jacobian of the system with respect to the coefficients of the collocation solution and with respect to the coefficients of the coordinate transform (optional).
        '''
        # write your jacobian wrt y and wrt transform here
        jac = None
        transform_jac = None
        return jac, transform_jac

    def bv_jacobian(self, y, transform):
        '''
        Calculate the jacobian of the boundary conditions with respect to the coefficients of the collocation solution and with respect to the coefficients of the coordinate transform (optional).
        '''
        # write your boundary condition jacobian wrt y and wrt transform here
        bv_jac = None
        bv_transform_jac = None
        return bv_jac, bv_transform_jac
