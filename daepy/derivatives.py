import numpy as np
from multiprocessing import Pool
import dill

def approx_jac(stream, x, epsilon, pool, *args, **kwargs):
    '''
    Compute a finite difference approximation of the jacobian of a serialised function *stream* at *x* with stepsize *epsilon* using a :class:`multiprocessing.pool.Pool` *pool*. Extra arguments and keyword arguments for the serialsed function may be passed following the other arguments. A python function can be serialised using :func:`dill.dumps`.
    '''
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
