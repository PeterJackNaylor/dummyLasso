
from .am import association_measure
import jax.numpy as np

def spearman_JAX(X, Y):
    rs = np.corrcoef(X, Y)
    if rs.shape == (2, 2):
        return rs[1, 0]
    else:
        return rs


def method(X, Y):
    rho = np.absolute(spearman_JAX(X, Y))
    # tau = kendalltau(X, Y).correlation
    # rho = spearmanr(X, Y).correlation
    return rho
    
pearson_correlation = association_measure(method)
