from typing import Union, List

import numpy as np
import tensorflow as tf
from scipy.special import comb, gegenbauer as scipy_gegenbauer

from gpflow.base import TensorType
from gpflow.config import default_float

from gpflux.vish.fundamental_set import FundamentalSystemCache
from gpflux.vish.gegenbauer_polynomial import Gegenbauer
from gpflux.vish.misc import surface_area_sphere


__all__ = [
    "num_harmonics",
    "eigenvalue_harmonics",
    "SphericalHarmonicsCollection",
    "FastSphericalHarmonicsCollection",
]


def eigenvalue_harmonics(
    degrees: Union[int, float, np.ndarray], dimension: int
) -> Union[int, float, np.ndarray]:
    """
    Eigenvalue of a spherical harmonic of a specific degree.

    :param degrees: a single, or a array of degrees
    :param dimension:
        S^{d-1} = { x ∈ R^d and ||x||_2 = 1 }
        For a circle d=2, for a ball d=3

    :return: the corresponding eigenvalue of the spherical harmonic
        for the specified degrees, same shape as degrees.
    """
    assert dimension >= 3, "We only support dimensions >= 3"

    return degrees * (degrees + dimension - 2)


def num_harmonics(dimension: int, degree: int) -> int:
    r"""
    Number of spherical harmonics of a particular degree n in
    d dimensions. Referred to as N(d, n).

    param dimension:
        S^{d-1} = { x ∈ R^d and ||x||_2 = 1 }
        For a circle d=2, for a ball d=3
    param degree: degree of the harmonic
    """
    if degree == 0:
        return 1
    elif dimension == 3:
        return int(2 * degree + 1)
    else:
        return int(
            np.round(
                (2 * degree + dimension - 2)
                / degree
                * comb(degree + dimension - 3, degree - 1)
            )
        )


class SphericalHarmonicsCollection:
    r"""
    Contains all the spherical harmonic levels up to a `max_degree`.
    Total number of harmonics in the collection is given by
    \sum_{degree=0}^max_degree num_harmonics(dimension, degree)
    """

    def __init__(
        self, dimension: int, degrees: Union[int, List[int]], debug: bool = False,
    ):
        """
        :param dimension: if d = dimension, then
            S^{d-1} = { x ∈ R^d and ||x||_2 = 1 }
            For a circle d=2, for a ball d=3
        :param degrees: list of degrees of spherical harmonic levels,
            if integer all levels (or degrees) up to `degrees` are used.
        highest degree of polynomial
            in the collection (exclusive)
        :param debug: print debug messages.
        """
        assert (
            dimension >= 3
        ), f"Lowest supported dimension is 3, you specified {dimension}"
        self.debug = debug

        if isinstance(degrees, int):
            degrees = list(range(degrees))

        self.fundamental_system = FundamentalSystemCache(dimension)
        self.harmonic_levels = [
            SphericalHarmonicsLevel(dimension, degree, self.fundamental_system)
            for degree in degrees
        ]

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=default_float())])
    def __call__(self, x: TensorType,) -> TensorType:
        """
        Evaluates each of the spherical harmonic level in the collection,
        and stacks the results.
        :param x: TensorType, [N, D]
            N points with unit norm in cartesian coordinate system.
        :return: [num harmonics in collection, N]
        """
        if self.debug:
            print("print: __call__ spherical harmonics")
            tf.print("tf.print: __call__ spherical harmonics")

        values = map(
            lambda harmonic: harmonic(x), self.harmonic_levels
        )  # List of length `max_degree` with Tensor [num_harmonics_degree, N]

        return tf.concat(list(values), axis=0)  # [num_harmonics, N]

    def __len__(self):
        return sum(len(harmonic_level) for harmonic_level in self.harmonic_levels)

    def num_levels(self):
        return len(self.harmonic_levels)

    def addition(self, X, X2=None):
        """For test purposes only"""
        return tf.reduce_sum(
            tf.stack(
                values=[level.addition(X, X2) for level in self.harmonic_levels],
                axis=0,
            ),
            axis=0,
        )  # [N1, N2]


class FastSphericalHarmonicsCollection(SphericalHarmonicsCollection):
    """
    Slightly faster implementation (approx 10%) of the `__call__` method than
    the one in `SphericalHarmonicsCollection` as we don't make use of a `map`.
    """

    def __init__(
        self, dimension: int, degrees: Union[int, List[int]], debug: bool = True
    ):
        """
        :param dimension: if d = dimension, then
            S^{d-1} = { x ∈ R^d and ||x||_2 = 1 }
            For a circle d=2, for a ball d=3
        :param degrees: list of degrees of spherical harmonic levels,
            if integer all levels (or degrees) up to `degrees` are used.
        :param debug: print debug messages.
        """
        super().__init__(dimension, degrees, debug)

        max_power = int(max(max(h.gegenbauer.powers) for h in self.harmonic_levels))
        self.num_harmonics = sum(len(h) for h in self.harmonic_levels)

        weights = np.zeros((self.num_harmonics, max_power + 1))  # [M, P]
        powers = np.zeros((self.num_harmonics, max_power + 1))  # [M, P]
        begin = 0
        for harmonic in self.harmonic_levels:
            coeffs = harmonic.gegenbauer.coefficients
            pows = harmonic.gegenbauer.powers
            for c, p in zip(coeffs, pows):
                for i in range(len(harmonic)):
                    weights[begin + i, int(p)] = c
                    powers[begin + i, int(p)] = p
            begin += len(harmonic)

        self.weights = tf.convert_to_tensor(weights)  # [M, P]
        self.powers = tf.convert_to_tensor(powers)  # [M, P]

        self.V = tf.convert_to_tensor(
            np.concatenate([harmonic.V for harmonic in self.harmonic_levels], axis=0)
        )  # [M, D]

        self.L_inv = tf.linalg.LinearOperatorBlockDiag(
            [
                tf.linalg.LinearOperatorFullMatrix(harmonic.L_inv)
                for harmonic in self.harmonic_levels
            ]
        )  # [M, M] block diagonal

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, None], dtype=default_float())]
    )
    def __call__(self, X: TensorType) -> TensorType:
        """
        Evaluates each of the spherical harmonics in the collection,
        and stacks the results.
        :param x: TensorType, [N, D]
            N points with unit norm in cartesian coordinate system.
        :return: [num harmonics in collection, N]
        """
        VXT = tf.matmul(self.V, X, transpose_b=True)  # [M, N, 1]
        tmp = self.weights[:, None, :] * (
            VXT[:, :, None] ** self.powers[:, None, :]
        )  # [M, N, P]
        gegenbauer_at_VXT = tf.reduce_sum(tmp, axis=-1)  # [M, N]
        return self.L_inv.matmul(gegenbauer_at_VXT)  # [M, N]


class SphericalHarmonicsLevel:
    r"""
    Spherical harmonics \phi(x) in a specific dimension and degree (or level).

    The harmonics are constructed by
    1) Building a fundamental set of directions {v_i}_{i=1}^N,
        where N is number of harmonics of the degree.
        Given these directions we have that {c(<v_i, x>)}_i is a basis,
        where c = gegenbauer(degree, alpha) and alpha = (dimension - 2)/2.
        See Definition 3.1 in [1].
    2) Using Gauss Elimination we orthogonalise this basis, which
       corresponds to the Gram-Schmidt procedure.

    [1] Approximation Theory and Harmonic Analysis on Spheres and Balls,
        Feng Dai and Yuan Xu, Chapter 1. Spherical Harmonics,
        https://arxiv.org/pdf/1304.2585.pdf
    """

    def __init__(self, dimension: int, degree: int, fundamental_system=None):
        r"""
        param dimension: if d = dimension, then
            S^{d-1} = { x ∈ R^d and ||x||_2 = 1 }
            For a circle d=2, for a ball d=3
        param degree: degree of the harmonic, also referred to as level.
        """
        assert (
            dimension >= 3
        ), f"Lowest supported dimension is 3, you specified {dimension}"
        self.dimension, self.degree = dimension, degree
        self.alpha = (self.dimension - 2) / 2.0
        self.num_harmonics_in_level = num_harmonics(self.dimension, self.degree)

        self.V = fundamental_system.load(self.degree)

        # surface area of S^{d−1}
        self.surface_area_sphere = surface_area_sphere(dimension)
        # normalising constant
        c = self.alpha * self.surface_area_sphere / (degree + self.alpha)
        VtV = np.dot(self.V, self.V.T)
        self.A = c * scipy_gegenbauer(self.degree, self.alpha)(VtV)
        self.L = np.linalg.cholesky(self.A)  # [M, M]
        # Cholesky inverse corresponds to the weights you get from Gram-Schmidt
        self.L_inv = np.linalg.solve(self.L, np.eye(len(self.L)))
        self.gegenbauer = Gegenbauer(self.degree, self.alpha)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, None], dtype=default_float())]
    )
    def __call__(self, X: TensorType) -> TensorType:
        r"""
        :param X: M normalised (i.e. unit) D-dimensional vector, [N, D]

        :return: `X` evaluated at the M spherical harmonics in the set.
            [\phi_m(x_i)], shape [M, N]
        """
        VXT = tf.matmul(self.V, X, transpose_b=True)  # [M, N]
        zonals = self.gegenbauer(VXT)  # [M, N]
        return tf.matmul(self.L_inv, zonals)  # [M, N]

    # TODO(Vincent) for some reason Optional[TensorType] doesn't work
    def addition(self, X: TensorType, Y: TensorType = None) -> TensorType:
        r"""
        Addition theorem. The sum of the product of all the spherical harmonics evaluated
        at x and x' of a specific degree simplifies to the gegenbauer polynomial evaluated
        at the inner product between x and x'.

        Mathematically:
            \sum_{k=1}^{N(dim, degree)} \phi_k(X) * \phi_k(Y)
                = (degree + \alpha) / \alpha / \omega_d * C_degree^\alpha(X^T Y)
        where \alpha = (dimension - 2) / 2 and omega_d the surface area of the
        hypersphere S^{d-1}.

        :param X: Unit vectors on the (hyper) sphere [N1, D]
        :param Y: Unit vectors on the (hyper) sphere [N2, D].
            If None, X is used as Y.

        :return: [N1, N2]
        """
        if Y is None:
            Y = X
        XYT = tf.matmul(X, Y, transpose_b=True)  # [N1, N2]
        c = self.gegenbauer(XYT)  # [N1, N2]
        return (
            (self.degree / self.alpha + 1.0) / self.surface_area_sphere * c
        )  # [N1, N2]

    def addition_at_1(self, X: TensorType) -> TensorType:
        r"""
        Evaluates \sum_k \phi_k(x) \phi_k(x), notice the argument at which we evaluate
        the harmonics is equal. See `self.addition` for the general case.

        This simplifies to
            \sum_{k=1}^{N(dim, degree)} \phi_k(x) * \phi_k(x)
                = (degree + \alpha) / \alpha / \omega_d * C_degree^\alpha(1)

        as all vectors in `X` are normalised so that x^\top x == 1.

        :param X: only used for it's X.shape[0], [N, D]
        :return: [N, 1]
        """
        c = (
            tf.ones((X.shape[0], 1), dtype=X.dtype) * self.gegenbauer.value_at_1
        )  # [N, 1]
        return (self.degree / self.alpha + 1.0) / self.surface_area_sphere * c  # [N, 1]

    def eigenvalue(self) -> float:
        """
        Spherical harmonics are eigenfunctions of the Laplace-Beltrami operator
        (also known as the Spherical Laplacian). We return the associated
        eigenvalue.

        The eigenvalue of the N(dimension, degree) number of spherical harmonics
        on the same level (i.e. same degree) is the same.
        """
        return eigenvalue_harmonics(self.degree, self.dimension)

    def __len__(self):
        return self.num_harmonics_in_level
