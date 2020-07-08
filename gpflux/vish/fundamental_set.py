from pathlib import Path

import numpy as np
from scipy import linalg, optimize
from scipy.special import gegenbauer

from gpflux.vish.memorize import memorize


@memorize(
    Path(__file__).parent.absolute() / ".fundamental_system_cache.json",
    lambda d, n, N: f"dimension_{d}_degree_{n}_numharmonics_{N}",
)
def build_fundamental_system(
    dimension, degree, num_harmonics, *, gtol=1e-5, num_restarts=1
):
    """
    TODO(Vincent): document what is happening!

    Based on [1, Defintion 3.1]

    [1] Approximation Theory and Harmonic Analysis on Spheres and Balls,
        Feng Dai and Yuan Xu, Chapter 1. Spherical Harmonics,
        https://arxiv.org/pdf/1304.2585.pdf
    """
    Z0 = np.eye(dimension)[-1]  # first direction is the north pole
    X_done = (Z0 / np.sqrt(np.sum(np.square(Z0)))).reshape(1, dimension)
    M_done = gegenbauer(degree, (dimension - 2.0) / 2.0)(1.0).reshape(1, 1)
    M_done_chol = np.sqrt(M_done)

    for i in range(1, num_harmonics):
        args = (X_done, M_done_chol, degree, dimension)
        Z_next, ndet = None, np.inf
        for _ in range(num_restarts):
            r = optimize.fmin_bfgs(
                f=_det_obj,
                x0=np.random.randn(dimension),
                fprime=_grad,
                args=args,
                full_output=True,
                gtol=gtol,
            )
            if r[1] < ndet:
                Z_next, ndet, *_ = r
            print(f"det: {-ndet}, ({i + 1} of {num_harmonics})")

        X_next = Z_next.reshape(1, dimension) / np.sqrt(np.sum(np.square(Z_next)))
        X_done = np.vstack([X_done, X_next])
        XtX = np.dot(X_done, X_done.T)
        M_done = gegenbauer(degree, (dimension - 2.0) / 2.0)(XtX)
        M_done_chol = np.linalg.cholesky(M_done)

    return X_done


def _det_obj(Z, X_done, M_done_chol, degree, dimension):
    """
    :param Z: is a potential vector for the next fundamental point (it will get normalized)
    :param X_done: is a matrix of existing fundamental points [num_done, D]
    :param M_done_chol: is the cholesky of the matrix M of the done points [num_done, num_done]

    :return: the negative-increment of the determinant of the matrix with Z (normalized) added
    to the done points
    """
    X = Z / np.sqrt(np.sum(np.square(Z)))
    XXd = np.dot(X_done, X)  # [num_done,]
    c = gegenbauer(degree, (dimension - 2.0) / 2.0)
    M_new = c(1.0)
    M_cross = c(XXd)
    # return -Mdet_done * (M_new - np.dot(np.dot(Mi_done, M_cross), M_cross))
    # return - np.dot(np.dot(Mi_done, M_cross), M_cross)
    tmp = linalg.solve_triangular(M_done_chol, M_cross, trans=0, lower=True)
    return np.sum(np.square(tmp)) - M_new


def _grad(Z, X_done, M_done_chol, degree, dimension):
    norm = np.sqrt(np.sum(np.square(Z)))
    X = Z / norm
    XXd = np.dot(X_done, X)  # [num_done,]
    c = gegenbauer(degree, (dimension - 2.0) / 2.0)
    M_cross = c(XXd)

    tmp = linalg.solve_triangular(M_done_chol, M_cross, trans=0, lower=True)
    # dM_cross = -2. * np.dot(Mi_done, M_cross)
    dM_cross = 2.0 * linalg.solve_triangular(M_done_chol, tmp, lower=True, trans=1)
    dXXd = c.deriv()(XXd) * dM_cross
    dX = np.dot(X_done.T, dXXd)
    dZ = (dX - X * np.dot(X, dX)) / norm

    return dZ
