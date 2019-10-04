from sys import path
path.insert(0, '..')
import numpy as np
from daepy.bvp import load_solution
from matplotlib import pyplot as plt

alpha = 5

sol = load_solution('test.npz')

l = np.linspace(0,1)
plt.plot(l, sol(l)[0])
plt.plot(l, l**(3+alpha), '--')

plt.legend(['Numerical solution', 'Analytical solution'])
plt.title('Save/load test')
plt.show()
