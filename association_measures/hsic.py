from association_measures.am import association_measure
import jax.numpy as np

from jax import vmap
from .kernel_tools import center, get_kernel_function


# def HSIC(X, Y, kernel="gaussian", normalised=False, sigma=None):
#     """
#     Computes the HSIC between X and Y with a given kernel
#     Parameters
#     ----------
#     X : numpy array like object where the rows correspond to the samples
#         and the columns to features.

#     Y : numpy array like, of same size as X and one single output.

#     kernel: string designating or distance, gaussian or linear

#     normalised: bool, wether or not to use the normalised HSIC
#         where HSICn = HSIC(X, Y) / (HSIC(X,X).HSIC(Y,Y)) **0.5

#     sigma: None or float, hyper parameter for the gaussian kernel.
#         If set to None, it takes sigma as the median of distance matrix.

#     Returns
#     -------
#     numpy array of size the number of input features of X
#     which holds the HSIC between each feature and Y.
#     """
#     n, d = X.shape
#     ny, dy = Y.shape

#     assert n == ny
#     assert dy == 1

#     kernel, kernel_params = get_kernel_function(kernel, nfeats=sigma)

#     Ky = center(kernel(Y[:, 0], **kernel_params))

#     if normalised:
#         hsic_yy = np.trace(np.matmul(Ky, Ky))

#     def compute(k):
#         Kx = center(kernel(X[:, k], **kernel_params))
#         hsic = np.trace(np.matmul(Kx, Ky))
#         if normalised:
#             hsic_xx = np.trace(np.matmul(Kx, Kx))
#             norm = (hsic_xx * hsic_yy) ** 0.5
#             hsic = hsic / norm
#         return hsic

#     hsic_stat = vmap(compute)(np.arange(d))

#     return hsic_stat

def precompute_kernels(X, kernel="gaussian", sigma=None):
    kernel, kernel_params = get_kernel_function(kernel, nfeats=sigma)
    Kx = center(kernel(X, **kernel_params))
    return Kx

def method_hsic(X, Y, precomp_K=None, kernel="gaussian", normalised=False, sigma=None, **args):


    # we could save some computation by saving Kx and Ky, because we could compute them
    # once instead of d*d. 
    if precomp_K is None:
        Kx = precompute_kernels(X, kernel=kernel, sigma=sigma)
        Ky = precompute_kernels(Y, kernel=kernel, sigma=sigma)
    else:
        Kx = precomp_K[0]
        Ky = precomp_K[1]
    hsic = np.trace(np.matmul(Kx, Ky))

    if normalised:
        hsic_xx = np.trace(np.matmul(Kx, Kx))
        hsic_yy = np.trace(np.matmul(Ky, Ky))
        norm = (hsic_xx * hsic_yy) ** 0.5
        hsic = hsic / norm
    return hsic

HSIC = association_measure(method_hsic)
