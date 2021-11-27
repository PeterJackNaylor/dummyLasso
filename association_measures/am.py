import jax.numpy as np
from tqdm import trange
from jax import vmap, pmap
from jax import jit

def perform_loop(func, X, Y, **args):
    n, d = X.shape
    if Y is not None:
        ny, nd = Y.shape

        assert n == ny
        assert nd == 1
    else:
        Y = X
        nd = d

    if nd == d:
        indices = np.triu_indices(d, k=0, m=nd)
    else:
        indices = np.triu_indices(d, k=-d, m=nd)

    #indices = np.stack(indices, axis=0)
    if 'precompute' in args.keys():
        if len(args['precompute']) == 1:
            Kx = args['precompute'][0]
            def func_with_indices(i,j):
                return func(X[:,i], Y[:,j], precomp_K=(Kx[i], Kx[j]), **args)
        else:
            Kx = args['precompute'][0]
            Ky = args['precompute'][1]
            def func_with_indices(i,j):
                return func(X[:,i], Y[:,j], precomp_K=(Kx[i], Ky[j]), **args)
    else:
        def func_with_indices(i,j):
            return func(X[:,i], Y[:,j], **args)

    result = vmap(func_with_indices)(indices[0], indices[1])


    # for idx in trange(size):
    #     i, j = indices[0][idx], indices[1][idx]
    #     stats = stats.at[i,j].set(func(X[:,i], Y[:,j]))


    if nd == d:
        result_r = np.zeros((d, nd))
        result_r = result_r.at[indices].set(result)

        i_lower = np.tril_indices(d, -1, m=nd)
        result_r = result_r.at[i_lower].set(result_r.T[i_lower])
        result = result_r
    
    return result

def association_measure(func):
    """
    The KnockOff object is a transformer from the sklearn
    base object.
    :param alpha: float between 0 and 1. Sets the FDR rate.
    :param measure_stat: string, sets the association measure

    The model parameters once fitted will be alpha_indices.
    """

    def am(X, Y=None, **args):
        output = perform_loop(func, X, Y, **args)
        return output
    return am
