from sys import path
path.insert(0, '..')
import numpy as np
from test_system import DAE
from daepy import BVP
from matplotlib import pyplot as plt

alpha = 5
dae = DAE(alpha)

degree = 3
intervals = 10
bvp = BVP(dae, degree, intervals)
bvp.initial_guess([lambda x: 0], initial_interval=[0,1])

sol = bvp.solve(method='nleqres', tol=1e-14, maxiter=100, disp=True)

l = np.linspace(0,1)
plt.plot(l, sol.eval(l))
plt.plot(l, l**(3+alpha), '--')

plt.legend(['Numerical solution', 'Analytical solution'])
plt.title('Basic usage example')
plt.show()

sol.save('test')
