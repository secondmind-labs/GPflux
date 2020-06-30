from pathlib import Path

import numpy as np
from scipy import linalg, optimize
from scipy.special import gegenbauer as Gegenbauer

from gpflux.vish.memorize import memorize


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
    alpha = (dimension - 2) / 2.0
    gegenbauer = Gegenbauer(degree, alpha)

    # 1. Choose first direction in system to be north pole
    Z0 = np.eye(dimension)[-1]
    X_system = normalize(Z0).reshape(1, dimension)
    M_system_chol = cholesky_of_gegenbauered_gram(gegenbauer, X_system)

    # 2. Find a new vector incrementally by max'ing the determinant of the gegenbauered Gram
    for i in range(1, num_harmonics):

        Z_next, ndet, restarts = None, np.inf, 0
        while restarts <= num_restarts:
            x_init = np.random.randn(dimension)
            result = optimize.fmin_bfgs(
                f=negative_det_objective,
                fprime=grad_negative_det_objective,
                x0=x_init,
                args=(X_system, M_system_chol, gegenbauer),
                full_output=True,
                gtol=gtol,
            )

            if result[1] <= ndet:
                Z_next, ndet, *_ = result
                break
            else:
                # Try again with new x_init
                restarts += 1

        X_next = normalize(Z_next).reshape(1, dimension)
        X_system = np.vstack([X_system, X_next])
        M_system_chol = cholesky_of_gegenbauered_gram(gegenbauer, X_system)

    return X_system


def negative_det_objective(Z, X_system, M_system_chol, gegenbauer):
    """
    :param Z: is a potential vector for the next fundamental point (it will get normalized)
    :param X_system: is a matrix of existing fundamental points [num_done, D]
    :param M_system_chol: is the cholesky of the matrix M of the done points [num_done, num_done]

    :return: the negative-increment of the determinant of the matrix with Z (normalized) added to the done points
    """
    X = normalize(Z)
    XXd = np.dot(X_system, X)  # [num_done,]

    M_new = gegenbauer(1.0)  # X normalized so X @ X^T = 1
    M_cross = gegenbauer(XXd)

    res = linalg.solve_triangular(M_system_chol, M_cross, trans=0, lower=True)
    return np.sum(np.square(res)) - M_new


def grad_negative_det_objective(Z, X_system, M_system_chol, gegenbauer):
    X = normalize(Z)
    XXd = np.dot(X_system, X)  # [num_done,]

    M_cross = gegenbauer(XXd)

    res = linalg.solve_triangular(M_system_chol, M_cross, trans=0, lower=True)
    dM_cross = 2.0 * linalg.solve_triangular(M_system_chol, res, trans=1, lower=True,)
    dXXd = gegenbauer.deriv()(XXd) * dM_cross
    dX = np.dot(X_system.T, dXXd)
    dZ = (dX - X * np.dot(X, dX)) / norm(Z)
    return dZ


def cholesky_of_gegenbauered_gram(gegenbauer_polynomial, x_matrix):
    XtX = x_matrix @ x_matrix.T
    return np.linalg.cholesky(gegenbauer_polynomial(XtX))


def normalize(vec):
    return vec / norm(vec)


def norm(vec):
    return np.sqrt(np.sum(np.square(vec)))
