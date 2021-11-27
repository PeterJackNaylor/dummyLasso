

from association_measures.hsic import precompute_kernels
import jax.numpy as np

import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import association_measures as am
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
from jax import jacfwd, jacrev
from jax.numpy import linalg
from penalties import penalty_dic
from functools import partial
from association_measures.kernel_tools import check_vector
from tqdm import trange

available_am = ["PC", "DC", "TR", "HSIC", "cMMD", "pearson_correlation"]
kernel_am = ["HSIC", "cMMD"]
precomputed_am = ["HSIC", "cMMD"]
available_kernels = ["distance", "gaussian", "linear"]

class DC_Lasso(BaseEstimator, TransformerMixin):
    """
    The KnockOff object is a transformer from the sklearn
    base object.
    :param alpha: float between 0 and 1. Sets the FDR rate.
    :param measure_stat: string, sets the association measure

    The model parameters once fitted will be alpha_indices.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        measure_stat: str = "PC",
        kernel: str = "linear",
        penalty: str = "None",
        normalise_input: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        assert measure_stat in available_am, "measure_stat incorrect"
        assert kernel in available_kernels, "kernel incorrect"
        self.measure_stat = measure_stat
        self.kernel = kernel
        self.normalise_input = normalise_input
        self.penalty = penalty
        self.precompute = measure_stat in precomputed_am

    def get_assoc(self, x, y=None, **kwargs):

        args = {}
        if self.measure_stat in kernel_am:
            args["kernel"] = self.kernel

        if self.normalise_input:
            x = x / np.linalg.norm(x, ord=2, axis=0)

        assoc_func = self.get_association_measure()

        return assoc_func(x, y, **args, **kwargs)

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        seed: int = 42,
    ):
        """Fits model in a supervised manner following algorithm 1 in the paper
        *Model-free Feature Screening and FDR Control with Knockoff Features*
        by Liu et Al (2021).
        If d < n2 / 2 we do not perform the screening set to reduce the data.

        Parameters
        ----------
        X : numpy array like object where the rows correspond to the samples
            and the columns to features.

        y : numpy array like, which can be multi-dimensional.

        n1 : float between 0 and 1. Screening is applied on n1 percentage of
        the initial dataset

        d : integer, sets the number of features to reduce the dataset in the
        screening step

        Returns
        -------
        Itself, to comply with sklearn rules.
        """

        if seed:
            key = random.PRNGKey(seed)

        X, y = check_X_y(X, y)
        X = np.asarray(X)
        n, p = X.shape

        ny, = y.shape
        assert n == ny
        y = check_vector(y)

        D = self.get_assoc

        if self.precompute:
            #mostly for HSIC and cMMD
            indicesX = np.arange(p)
            indicesY = np.arange(1)
            jit_precompute_kernels = jit(precompute_kernels)
            @jit
            def precompX(k):
                return jit_precompute_kernels(X[:, k])
            def precompY(k):
                return jit_precompute_kernels(y[:, k])
            Kx = vmap(precompX)(indicesX)
            Ky = vmap(precompY)(indicesY)
            Dxy = D(X, y, precompute=(Kx, Ky))
            Dxx = D(X, precompute=(Kx,))
        else:
            # D = jit(D)
            Dxy = D(X, y)
            Dxx = D(X)


        def formula(theta):
            return (theta * Dxy).sum() + 0.5 * (theta * (Dxx * theta).T).sum() + self.penalty_func(theta)

        minfunc = vmap( formula )
        J = jacfwd(formula)
        def minJacobian(x): return x - 0.000001*J(x)  
        theta = random.uniform(key, shape=(p,), dtype='float32',minval=.0, maxval=5.0)

        vfuncHS = vmap(minJacobian)
        for epoch in trange(150):
            theta = vfuncHS(theta)
            # positivity constraint
            theta = theta.clip(0)

        minimums = minfunc(theta)
        import pdb; pdb.set_trace()
        return self

    def fit_transform(self, X, y, **fit_params):
        """Fits and transforms an input dataset X and y.

        Parameters
        ----------
        X : numpy array like object where the rows correspond to the samples
            and the columns to features.

        y : numpy array like, which can be multi-dimensional.

        Returns
        -------
        The new version of X corresponding to the selected features.
        """

        return self.fit(X, y, **fit_params).transform(X, y)

    def get_association_measure(self):
        """Returns the correct association measure
        given the attribute in __init__.
        """
        if self.measure_stat == "PC":
            f = am.projection_corr
        elif self.measure_stat == "TR":
            f = am.tr
        elif self.measure_stat == "HSIC":
            f = am.HSIC
        elif self.measure_stat == "cMMD":
            f = am.cMMD
        elif self.measure_stat == "DC":
            f = am.distance_corr
        elif self.measure_stat == "pearson_correlation":
            f = am.pearson_correlation
        else:
            error_msg = f"associative measure undefined {self.measure_stat}"
            raise ValueError(error_msg)
        return f

    def penalty_func(self, theta):
        return penalty_dic[self.penalty](theta)

    def _more_tags(self):
        return {"stateless": True}
