import abc
from typing import List, Union, Optional, Tuple

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.base import Parameter, TensorType, TensorType
from gpflow.utilities import positive

from gpflux.vish.gegenbauer_polynomial import Gegenbauer
from gpflux.vish.matern_spectral_density import spectral_density
from gpflux.vish.misc import surface_area_sphere
from gpflux.vish.spherical_harmonics import eigenvalue_harmonics
from gpflux.vish.trigonometric_integrals import integrate_0_pi__J1_cos_k_sin_d


class ZonalKernel(gpflow.kernels.Kernel, metaclass=abc.ABCMeta):
    r"""
    Type of kernel k(x, x') for which there exists a shape function s(.)
    such that k(x, x') = s(x^T x'). These kernels play a crucial role as
    their Kernel operator K shares the same eigen functions as the
    Laplace-Beltrami operator, i.e. the spherical harmonics.

    Using Mercer's theorem we can decompose the kernel as a sum
    over it's eigenfunctions,
            k(x, x') = \sum_i \lambda_i \phi_i(x) \phi_i(x').
    where K \phi_i(.) = \lambda_i \phi_i(.),
    and the kernel operator K defined as (K f)(x) = \int k(x', x) f(x') dx'.
    """

    def __init__(
        self,
        dimension: int,
        truncation_level: int,
        variance: float = 1.0,
        weight_variances: Union[float, List[float]] = 1.0,
        name: str = None,
    ):
        r"""
        :param truncation_level: we decompose the kernel as
            k(x, x') = \sum_\ell \lambda_\ell \phi_\ell(x) \phi_\ell(x').
            This argument determines how many components we use.
        """
        assert (
            dimension >= 3
        ), f"Lowest supported dimension is 3, you specified {dimension}"

        super().__init__(name=name)
        self.dimension = dimension
        self.alpha = (self.dimension - 2.0) / 2.0
        self.variance = Parameter(variance, transform=positive())
        self.weight_variances = Parameter(weight_variances, transform=positive())
        self.truncation_level = truncation_level  # referred to as L
        # surface area of S^{d−1}
        self.surface_area_sphere = surface_area_sphere(dimension)

        self.degrees = np.arange(truncation_level).reshape(-1, 1)  # [L, 1]
        self.gegenbauers = [Gegenbauer(level, self.alpha) for level in self.degrees]

    def get_first_num_eigenvalues_and_degrees(
        self, num: Optional[int] = None
    ) -> Tuple[TensorType, TensorType]:
        r"""
        First (largest) `num` eigenvalues \lambda_\ell of the kernel and
        the corresponding degree of the harmonic are returned. The eigenvalues
        are sorted from lower degree (starting with the constant harmonic (degree=0))
        to larger degree.

        :param num: number of first eigenvalues to return.
            If None, all available eigenvalues are given, i.e. up to
            `self.truncation_level`.

        :return: Array of eigenvalues and degrees, ([num, 1], [num, 1])
        """
        if num is None:
            num = len(self.degrees)

        return self.eigenvalues[:num], self.degrees[:num]

    @property
    def eigenvalues(self) -> TensorType:
        """
        :return: [L, 1]
        """
        return self.variance * self._eigenvalues_without_variance

    def K(self, X, X2=None):
        r"""
        Computes K using Mercer's theorem
            k(x, x') = \sum_i \lambda_i \phi_i(x) \phi_i(x').

        Notice that in the case of spherical harmonics \sum_i is
        actually \sum_level \sum_h, where h iterates over all the
        harmonics in he level. For zonal kernels the eigenvalue
        only depends on the level, which means that we can reduce
        the sum over the harmonics within the level to a single
        evaluation of the Gegenbauer, using the addition theorem.
        See SphericalHarmonicsLevel.addition for detailed explanation.

        :param X: vectors need to be normalised [N1, D]
        :param X2: vectors need to be normalised [N2, D]
        """
        if X2 is None:
            X2 = X

        inner_product = tf.matmul(X, X2, transpose_b=True)  # [N1, N2]
        cs = tf.stack(
            values=[c(inner_product) for c in self.gegenbauers], axis=0
        )  # [L, N1, N2]
        factors = self.degrees / self.alpha + 1.0  # [L, 1]
        sum_level_phi_X_phi_X2 = factors[..., None] * cs  # [L, N1, N2]
        # eigenvalues
        Lambda_level = self.eigenvalues[..., None]  # [L, 1, 1]
        return tf.reduce_sum(Lambda_level * sum_level_phi_X_phi_X2, axis=0)  # [N1, N2]

    def K_diag(self, X):
        """
        See `K` for more details.
        """
        ones = tf.ones((tf.shape(X)[0], 1), dtype=X.dtype)  # [N, 1]
        cs_at_one = tf.stack(
            values=[ones * c.value_at_1 for c in self.gegenbauers], axis=0
        )  # [L, N, 1]
        factors = self.degrees / self.alpha + 1.0  # [L, 1]
        sum_level_phi_X_phi_X2 = factors[..., None] * cs_at_one  # [L, N, 1]
        Lambda_level = self.eigenvalues[..., None]  # [L, 1, 1]
        return tf.reduce_sum(Lambda_level * sum_level_phi_X_phi_X2, axis=0)[:, 0]  # [N]


class Matern(ZonalKernel):
    def __init__(
        self,
        dimension: int,
        truncation_level: int,
        nu: float = 0.5,
        variance: float = 1.0,
        weight_variances: Union[float, List[float]] = 1.0,
        name: str = None,
    ):
        """
        Matern covariance on the (hyper)sphere.

        :param dimension:
            for the sphere in R^3 `dimension` is 2.
        :param nu:
            Specifies the continuity of the matern kernel. Typical values are nu = 1/2,
            and nu = 3/2. For nu = np.inf we end up with the SE kernel.
        """
        super().__init__(
            dimension=dimension,
            truncation_level=truncation_level,
            variance=variance,
            weight_variances=weight_variances,
            name=name,
        )
        self.nu = nu
        self.eigenvalue_harmonics = eigenvalue_harmonics(
            self.degrees, self.dimension
        )  # [N, L]
        self._eigenvalues_without_variance = spectral_density(
            s=np.sqrt(self.eigenvalue_harmonics),
            nu=self.nu,
            dimension=self.dimension,
            variance=1.0,
        )  # [L, 1]


class Parameterised(ZonalKernel):
    """Spectral Mixture and Minecraft style kernel learning the eigenvalues of the kernel"""

    def __init__(
        self,
        dimension: int,
        truncation_level: int,
        variance=1.0,
        weight_variances=1.0,
        name=None,
    ):
        super().__init__(
            dimension=dimension,
            truncation_level=truncation_level,
            variance=variance,
            weight_variances=weight_variances,
            name=name,
        )
        self._eigenvalues_without_variance = Parameter(
            np.ones((truncation_level, 1), dtype=gpflow.config.default_float()),
            transform=positive(),
        )  # [L, 1]


class ArcCosine(ZonalKernel):
    r"""
    ArcCosine kernel k(x, x') = sin(\theta) + (\theta - \pi) cos(\theta),
    where \theta is the angle between x and x'. For x and x`, \theta = x^T x'.

    We compute the eigenvalues of this kernel numerically using Funk-Hecke.
    """

    def __init__(
        self,
        dimension: int,
        variance=1.0,
        truncation_level=30,
        weight_variances=1.0,
        name=None,
    ):
        super().__init__(
            dimension=dimension,
            truncation_level=truncation_level,
            variance=variance,
            weight_variances=weight_variances,
            name=name,
        )
        all_eigenvalues_wo_variance = self._compute_eigenvalues()  # [L, 1]
        mask = all_eigenvalues_wo_variance > 1e-6
        # Due to the symmetric shape of the shape function of the ArcCosine kernel
        # all odd degrees larger than 3 have a zero eigenvalue. They get masked out.
        # mask = [False], [False], [False], [ True], [False], [ True], [False], ...

        # Only keep the degrees and eigenvalues that are non-zero
        self.degrees = self.degrees[mask].reshape(-1, 1)
        self._eigenvalues_without_variance = all_eigenvalues_wo_variance[mask].reshape(
            -1, 1
        )
        # The gegenbauers are not needed anymore for the ArcCosine kernel
        # because the analytical expression for k(x, x') is known. The
        # default (super) implementations of K and K_diag are also overwritten
        # with this expression.
        # However, for test purposes (see tests/test_kernels:test_arccosine),
        # we keep the list of gegenbauers and update it corresponding to the mask.
        self.gegenbauers = [Gegenbauer(level, self.alpha) for level in self.degrees]

    def _J(self, theta):
        return tf.sin(theta) + (np.pi - theta) * tf.cos(theta)

    def K_diag(self, X):
        ones = tf.ones((tf.shape(X)[0]), dtype=X.dtype)
        return self.variance * np.pi * ones

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        cos_theta = tf.matmul(X, X2, transpose_b=True)  # [N1, N2]
        jitter = 1e-15
        theta = tf.acos(jitter + (1 - 2 * jitter) * cos_theta)
        return self.variance * self._J(theta)

    def _compute_eigenvalues(self) -> np.ndarray:
        r"""
        Computes the coefficients for the angular kernel J1(t) = sin(t) + (pi - t) cos(t)
        following eq 2.11, theorem 2.9, page 9, Spherical Harmonics, Feng Dai and Yuan Xu

        for l = (dimension - 2) / 2, n = degree, and C gegenbauer polynomial
        omega_{d - 1} / C_n^l(1) \int J1(t) C_n^l(t) * (1 - t^2)^{(d - 3) / 2} dt
        """
        surface_area_sphere_d_minus_1 = surface_area_sphere(self.dimension - 1)

        eigenvalues = []
        for gegenbauer in self.gegenbauers:

            lambda_J1 = 0
            # We use the fact that the gegenbauer can be written as a polynomial in t
            for c, p in zip(gegenbauer.coefficients, gegenbauer.powers):
                lambda_J1 += c * integrate_0_pi__J1_cos_k_sin_d(
                    int(p), self.dimension - 2
                )

            gegenbauer_n_1 = gegenbauer.value_at_1
            eigenvalues.append(
                surface_area_sphere_d_minus_1 / gegenbauer_n_1 * lambda_J1
            )

        return np.array(eigenvalues).reshape(-1, 1)
