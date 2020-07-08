import pytest
import numpy as np
import tensorflow as tf

import gpflow
from gpflow.base import TensorLike
from gpflow.covariances import Kuf, Kuu
from gpflow.conditionals import conditional
from gpflow.kullback_leiblers import prior_kl
from gpflux.vish.conditional import Lambda_diag_elements
from gpflux.vish.inducing_variables import SphericalHarmonicInducingVariable
from gpflux.vish.kernels import ArcCosine, ZonalKernel
from gpflux.vish.spherical_harmonics import SphericalHarmonicsCollection


class NaiveSphericalHarmonicInducingVariable(SphericalHarmonicInducingVariable):
    """
    Wrapper for `SphericalHarmonicInducingVariable` to redirect the multiple-dispatch
    methods Kuu, Kuf, conditional and prior_kl to do the naive thing.
    """

    pass


@Kuu.register(NaiveSphericalHarmonicInducingVariable, ZonalKernel)
def Kuu_naive(
    inducing_variable: NaiveSphericalHarmonicInducingVariable,
    kernel: ZonalKernel,
    *,
    jitter=0.0,
):
    Lambda = Lambda_diag_elements(inducing_variable, kernel)  # [M]
    # Return diagonal matrix Kuu as a full (dense) matrix - which makes it 'naive'
    return tf.linalg.diag(Lambda)  # [M, M]


@Kuf.register(NaiveSphericalHarmonicInducingVariable, ZonalKernel, TensorLike)
def Kuf_naive(
    inducing_variable: NaiveSphericalHarmonicInducingVariable,
    kernel: ZonalKernel,
    Xnew,
):
    Lambda = Lambda_diag_elements(inducing_variable, kernel)  # [M]
    Phi = inducing_variable(Xnew)  # [M, N]
    return Lambda[:, None] * Phi  # [M, N]


@conditional.register(
    object, NaiveSphericalHarmonicInducingVariable, ZonalKernel, object
)
def conditional_naive(*args, **kwargs):
    naive_conditional_implementation = conditional.dispatch(
        object,
        gpflow.inducing_variables.inducing_variables.InducingVariables,
        gpflow.kernels.base.Kernel,
        object,
    )
    return naive_conditional_implementation(*args, **kwargs)


@prior_kl.register(NaiveSphericalHarmonicInducingVariable, ZonalKernel, object, object)
def prior_kl_naive(*args, **kwargs):
    naive_prior_kl_implementation = prior_kl.dispatch(
        gpflow.inducing_variables.inducing_variables.InducingVariables,
        gpflow.kernels.base.Kernel,
        object,
        object,
    )
    return naive_prior_kl_implementation(*args, **kwargs)


def get_data(num_data, dimension):
    X = np.random.randn(num_data, dimension)
    X = X / np.sum(X ** 2, axis=-1, keepdims=True) ** 0.5
    Y = np.sin(X[:, :1])
    return (X, Y)


@pytest.mark.parametrize("dimension", [3, 4])
@pytest.mark.parametrize("max_degree", [5, 6])
def test_equality(dimension, max_degree):
    data = get_data(100, dimension)

    # Naive model using dense Kuu matrices
    kernel = ArcCosine(dimension, truncation_level=max_degree)
    degrees = kernel.degrees
    harmonics = SphericalHarmonicsCollection(dimension, degrees=degrees)
    inducing_variable_naive = NaiveSphericalHarmonicInducingVariable(harmonics)

    model_naive = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=inducing_variable_naive,
        whiten=False,
    )
    q_sqrt_init = np.diag(
        Lambda_diag_elements(inducing_variable_naive, kernel).numpy() ** 0.5
    )  # [M, M]
    q_mu_init = q_sqrt_init @ np.random.randn(len(q_sqrt_init), 1)  # [M, 1]
    model_naive.q_sqrt.assign(q_sqrt_init[None])
    model_naive.q_mu.assign(q_mu_init)

    # 'Fast' model, leveraging the diagonal Kuu.
    inducing_variable = SphericalHarmonicInducingVariable(harmonics)
    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=inducing_variable,
        whiten=False,
    )
    model.q_sqrt.assign(q_sqrt_init[None])
    model.q_mu.assign(q_mu_init)

    np.testing.assert_allclose(model.elbo(data).numpy(), model_naive.elbo(data).numpy())

    # ELBO = Datafit - KL
    # If the ELBOs and the KLs are equal we know the Datafits are
    # equal as well.
    np.testing.assert_allclose(model.prior_kl().numpy(), model_naive.prior_kl().numpy())
