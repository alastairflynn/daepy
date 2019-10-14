from sys import path
path.insert(0, '..')
import numpy as np
from test_system import DAE
from daepy import BVP
from matplotlib import pyplot as plt

alpha = 10
dae = DAE(alpha)

degree = 3
intervals = 20
bvp = BVP(dae, degree, intervals)
bvp.initial_guess([lambda x: 0], initial_interval=[0,1])

def callback(p, sol):
    colour = (min((p-10)/40, 1.0), 0.0, max(1-(p-10)/40, 0.0))
    l = np.linspace(0,1)
    plt.plot(sol.forward(l), sol(l), color=colour) # plot using internal coordinate for smoother lines

steps = list(range(15,51,5))
bvp.continuation(alpha, method='pseudo_arclength', steps=steps, tol=1e-14, maxiter=100, disp=True, callback=callback)

plt.legend([r'$\alpha = $' + str(s) for s in [alpha] + steps])
plt.title('Parameter continuation example')
plt.show()
