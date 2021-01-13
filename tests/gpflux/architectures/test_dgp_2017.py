import numpy as np
import pytest

from gpflux.architectures.dgp_2017 import Config as DGP2017Config
from gpflux.architectures.dgp_2017 import build_deep_gp_2017
from gpflux.helpers import make_dataclass_from_class


class DemoConfig:
    num_inducing = 7
    inner_layer_qsqrt_factor = 1e-3
    between_layer_noise_variance = 1e-3
    likelihood_noise_variance = 1e-2
    white = True


@pytest.mark.parametrize(
    "layer_dims", [(3, 1, 1), (3, 3, 1), (3, 5, 1), (3, 1, 1, 1), (3, 2, 3, 1), (3, 5, 5, 1)],
)
@pytest.mark.parametrize("num_data", [1, 3, 5])
def test_smoke_build_deep_gp_2017(layer_dims, num_data):
    if layer_dims == (3, 2, 3, 1) and num_data == 1:
        pytest.skip("not working")

    config = make_dataclass_from_class(DGP2017Config, DemoConfig)
    X = np.random.randn(num_data, layer_dims[0])
    model = build_deep_gp_2017(X, layer_dims, config)
    _ = model(X, training=True)
    _ = model(X, training=False)
    _ = model.predict_y(X)
