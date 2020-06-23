import pytest
import numpy as np

from vish.helpers import (
    get_num_inducing,
    get_max_degree_closest_but_smaller_than_num_inducing,
)
from vish.kernels import ArcCosine, Matern
from vish.inducing_variables import (
    SphericalHarmonicInducingVariable,
    SphericalHarmonicsCollection,
)
from gpflux.vish.spherical_harmonics import num_harmonics
from gpflux.vish.conditional import Lambda_diag_elements


@pytest.mark.parametrize("dimension", [3, 8, 10])
@pytest.mark.parametrize("max_degree", [4, 5])
@pytest.mark.parametrize("kernel_type", ["matern", "arccosine"])
def test_number_of_inducing(dimension, max_degree, kernel_type):
    if kernel_type == "arccosine":
        kernel = ArcCosine(dimension, truncation_level=max_degree,)
        degrees = kernel.degrees
    elif kernel_type == "matern":
        kernel = Matern(dimension, truncation_level=max_degree,)
        degrees = kernel.degrees[:max_degree]

    harmonics = SphericalHarmonicsCollection(dimension, degrees=degrees)
    inducing_variable = SphericalHarmonicInducingVariable(harmonics)
    Lambda = Lambda_diag_elements(inducing_variable, kernel).numpy()
    assert len(Lambda) == get_num_inducing(kernel_type, dimension, max_degree)


@pytest.mark.parametrize("kernel_type", ["matern", "arccosine"])
@pytest.mark.parametrize("dimension", [3, 8, 10])
@pytest.mark.parametrize("num_inducing", [512, 1024, 2048])
def test_max_degree(kernel_type, dimension, num_inducing):
    (
        max_degree,
        num_inducing,
    ) = get_max_degree_closest_but_smaller_than_num_inducing(
        kernel_type, dimension, num_inducing
    )
    assert get_num_inducing(kernel_type, dimension, max_degree) <= num_inducing
    assert (
        get_num_inducing(kernel_type, dimension, max_degree + 1) > num_inducing
    )
