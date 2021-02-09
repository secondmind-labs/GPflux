import numpy as np
import pytest

from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from gpflux.helpers import make_dataclass_from_class


class DemoConfig:
    num_inducing = 7
    inner_layer_qsqrt_factor = 1e-3
    between_layer_noise_variance = 1e-3
    likelihood_noise_variance = 1e-2
    white = True


@pytest.mark.parametrize("input_dim", [7, 11])
@pytest.mark.parametrize("num_layers", [3, 5])
def test_smoke_build_constant_input_dim_deep_gp(input_dim, num_layers):
    config = make_dataclass_from_class(Config, DemoConfig)
    X = np.random.randn(13, input_dim)
    Y = np.random.randn(13, 1)
    model = build_constant_input_dim_deep_gp(X, num_layers, config)
    model.compile()
    model.fit(X, Y, epochs=1)
    _ = model(X)
