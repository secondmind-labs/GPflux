import os
import pytest

import numpy as np

import gpflow
from gpflux.vish.inducing_variables import SphericalHarmonicInducingVariable
from gpflux.vish.kernels import Matern, Parameterised, ArcCosine
from gpflux.vish.spherical_harmonics import SphericalHarmonicsCollection
from gpflux.vish.helpers import preprocess_data


def get_snelson():
    path = os.path.join(os.path.dirname(__file__), "snelson1d.npz")
    data = np.load(path)
    return (data["X"], data["Y"])


@pytest.fixture
def data():
    data, _, _ = preprocess_data(get_snelson())
    return data


def build_model(dimension, kernel_type, max_degree):
    truncation_level = 40

    if kernel_type == "Matern":
        kernel = Matern(dimension, truncation_level=truncation_level, nu=0.5)
        degrees = range(max_degree)
    elif kernel_type == "Param":
        kernel = Parameterised(dimension, truncation_level=truncation_level,)
        degrees = range(max_degree)
    elif kernel_type == "ArcCosine":
        kernel = ArcCosine(dimension)
        index_max_degree = np.argmin(abs(kernel.degrees - max_degree))
        degrees = kernel.degrees[: (index_max_degree + 1)]

    harmonics = SphericalHarmonicsCollection(dimension, degrees=degrees)
    inducing_variable = SphericalHarmonicInducingVariable(harmonics)

    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=inducing_variable,
        whiten=False,
    )
    return model


@pytest.mark.parametrize("kernel_type", ["Param", "Matern", "ArcCosine"])
def test_model(data, kernel_type):
    max_degree = 25
    model = build_model(data[0].shape[1], kernel_type, max_degree=max_degree)
    elbo1 = model.elbo(data)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        model.training_loss_closure(data),
        model.trainable_variables,
        options=dict(maxiter=30, disp=1),
    )

    elbo2 = model.elbo(data)
    assert elbo2 > elbo1


def test_increasing_elbo(data):
    """
    Check that elbo becomes tighter with increasing number of
    basis functions in the approximation.
    """
    elbos = []
    for max_degree in [5, 10, 20]:
        dimension = data[0].shape[1]
        model = build_model(dimension, "Matern", max_degree=max_degree)

        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            model.training_loss_closure(data),
            model.trainable_variables,
            options=dict(maxiter=30, disp=1),
        )
        elbos.append(model.elbo(data).numpy())

    assert all(elbos[i] <= elbos[i + 1] for i in range(len(elbos) - 1))
