import jax.numpy as np
from scipy.stats import kendalltau, spearmanr
from tqdm import trange
from .am import association_measure

def kendall_JAX(X, Y):
    pass

def spearman_JAX(X, Y):
    rs = np.corrcoef(X, Y)
    if rs.shape == (2, 2):
        return rs[1, 0]
    else:
        return rs

def method(X, Y):
    tau = X.sum()
    rho = spearman_JAX(X, Y)
    # tau = kendalltau(X, Y).correlation
    # rho = spearmanr(X, Y).correlation
    return 3 * tau - 2 * rho

tr = association_measure(method)

if __name__ == "__main__":
    x, y = [1, 2, 3, 4, 5], [5, 6, 7, 8, 7]
    print("rho: ", spearmanr(x, y).correlation)
    print("tau: ", kendalltau(x, y).correlation)
    print(f"{tr(x,y)=}")
