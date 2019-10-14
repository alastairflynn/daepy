from sys import path
path.insert(0, '..')
import numpy as np
from daepy import BVP, load_solution
from test_system import DAE
from matplotlib import pyplot as plt

alpha = 5
dae = DAE(alpha)

degree = 3
intervals = 10
bvp = BVP(dae, degree, intervals)

sol = load_solution('test.npz')
bvp.initial_solution(sol)

print('Residual:', np.linalg.norm(bvp.eval(bvp.state())))

l = np.linspace(0,1)
plt.plot(l, sol.eval(l))
plt.plot(l, l**(3+alpha), '--')

plt.legend(['Numerical solution', 'Analytical solution'])
plt.title('Save/load test')
plt.show()
