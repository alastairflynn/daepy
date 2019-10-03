class DAETemplate():
    '''
    The user must write a class with this template to define the BVP. The class must define the attributes *N*, *dindex* and *aindex* and the methods :meth:`fun`, :meth:`bv`. It may optionally define the methods :meth:`jacobian`, :meth:`bv_jacobian`, :meth:`update_parameter` and :meth:`parameter_jacobian`. You can of course add your own attributes and methods as well.
    '''
    def __init__(self, parameter):
        self.N = 2
        self.dindex = [0]
        self.aindex = [1]
        self.parameter = parameter

    def fun(self, x, y):
        '''
        Evaluate the system at points *x* where *y* is a :class:`.BVPSolution`.
        '''
        # write your system here
        residual = None
        return residual

    def bv(self, y):
        '''
        Evaluate the boundary conditions where *y* is a :class:`.BVPSolution`.
        '''
        # write your boundary conditions here
        bc = None
        return bc

    def jacobian(self, x, y):
        '''
        Calculate the jacobian of the system evaluated at points *x* with respect to the coefficients of the collocation solution and with respect to the coefficients of the coordinate transform where *y* is a :class:`.BVPSolution`.
        '''
        # write your jacobian wrt y and wrt transform here
        jac = None
        transform_jac = None
        return jac, transform_jac

    def bv_jacobian(self, y):
        '''
        Calculate the jacobian of the boundary conditions with respect to the coefficients of the collocation solution and with respect to the coefficients of the coordinate transform where *y* is a :class:`.BVPSolution`.
        '''
        # write your boundary condition jacobian wrt y and wrt transform here
        bv_jac = None
        bv_transform_jac = None
        return bv_jac, bv_transform_jac

    def update_parameter(self, p):
        '''
        Update the parameter. This is used for parameter continuation.
        '''
        self.parameter = p

    def parameter_jacobian(self, x, y):
        '''
        Calculate the derivative of the system with respect to the parameter where *y* is a :class:`.BVPSolution`. This is used for parameter continuation.
        '''
        param_jac = None
        bv_param_jac = None
        return param_jac, bv_param_jac
