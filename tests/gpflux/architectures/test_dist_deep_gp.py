import numpy as np
import pytest
import tensorflow as tf

from gpflux.architectures import DistConfig, build_dist_deep_gp
from gpflux.helpers import make_dataclass_from_class


class DemoConfig:
    num_inducing = 7
    inner_layer_qsqrt_factor = 1e-3
    hidden_layer_size = 2
    type_likelihood = "Gaussian"
    dim_output = 1
    likelihood_noise_variance = 1e-2
    whiten = True


@pytest.mark.parametrize("input_dim", [7])
@pytest.mark.parametrize("num_layers", [3])
def test_smoke_build_dist_deep_gp(input_dim, num_layers):
    config = make_dataclass_from_class(DistConfig, DemoConfig)
    X = np.random.randn(13, input_dim)
    Y = np.random.randn(13, 1)
    dgp = build_dist_deep_gp(X, num_layers, config)
    model_train = dgp.as_training_model()
    model_train.compile("Adam")
    model_train.fit((X, Y), epochs=1)
    model_test = dgp.as_prediction_model()
    _ = model_test(X)
