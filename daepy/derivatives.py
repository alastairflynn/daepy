import numpy as np
import dill
from multiprocessing import Pool

def approx_jac(stream, x, epsilon, pool, *args, **kwargs):
    fun = dill.loads(stream)
    cols = x.shape[0]
    current = fun(x, *args, **kwargs)
    rows = current.shape[0]
    jac = np.zeros((rows, cols))

    arguments = [(stream, x, epsilon, c, current, *args, *kwargs) for c in range(cols)]
    result = pool.starmap_async(step, arguments, chunksize=16)
    results = result.get()
    for r in results:
        jac[:,r[0]] = r[1]

    return jac

def step(stream, x, epsilon, c, current, *args, **kwargs):
    fun = dill.loads(stream)

    x[c] += epsilon
    f_step = fun(x, *args, **kwargs)
    x[c] -= epsilon
    return c, (f_step - current) / epsilon
