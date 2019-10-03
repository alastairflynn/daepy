from sys import path
path.insert(0, '..')
import numpy as np
from test_137 import DAE
from daepy import BVP
from matplotlib import pyplot as plt

degree = 3
intervals = 10

alpha = 5
dae = DAE(alpha)

bvp = BVP(dae, degree, intervals)
bvp.initial_guess([lambda x: 0*x, lambda x: 0*x], initial_interval=[0,1])

sol = bvp.solve(method='nleqres', tol=1e-14, maxiter=1000, disp=True)

l = np.linspace(0,1)
plt.plot(l, l**(3+alpha), '--')
plt.plot(l, sol(l)[0])

plt.legend(['Numerical solution', 'Analytical solution'])
plt.show()
