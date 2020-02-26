import numpy as np
import pytest

from gpflow.kernels import Matern52
from gpflow.mean_functions import Zero
from gpflow.likelihoods import Bernoulli, Beta, Gaussian, Poisson

from gpflux.layers import GPLayer, LikelihoodLayer
from gpflux.initializers import ZeroOneInitializer
from gpflux.helpers import construct_basic_kernel, construct_basic_inducing_variables

TEST_GPFLOW_LIKELIHOODS = [Bernoulli, Beta, Gaussian, Poisson]


def setup_gp_layer_and_data(num_inducing: int, **gp_layer_kwargs):
    input_dim = 30
    output_dim = 5
    num_data = 100
    data = make_data(input_dim, output_dim, num_data=num_data)

    kernel = construct_basic_kernel(Matern52(), output_dim)
    inducing_vars = construct_basic_inducing_variables(
        num_inducing, input_dim, output_dim
    )
    initializer = ZeroOneInitializer()
    mean_function = Zero(output_dim)

    gp_layer = GPLayer(
        kernel,
        inducing_vars,
        num_data,
        initializer=initializer,
        mean_function=mean_function,
        **gp_layer_kwargs
    )
    return gp_layer, data


def make_data(input_dim: int, output_dim: int, num_data: int):
    lim = [0, 20]
    sigma = 0.1

    X = np.random.random(size=(num_data, input_dim)) * lim[1]
    cov = Matern52()(X) + np.eye(num_data) * sigma ** 2
    Y = [
        np.random.multivariate_normal(np.zeros(num_data), cov)[:, None]
        for _ in range(output_dim)
    ]
    Y = np.hstack(Y)
    return X, Y


@pytest.mark.parametrize("GPFlowLikelihood", TEST_GPFLOW_LIKELIHOODS)
def test_call_shapes(GPFlowLikelihood):
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)
    likelihood_layer = LikelihoodLayer(GPFlowLikelihood())

    # 1. Run tests with gp layer outputting f_mean, f_var
    gp_layer.returns_samples = False

    f_mean, f_var = gp_layer(X)
    y_mean, y_var = likelihood_layer((f_mean, f_var), training=False)

    assert y_mean.shape == f_mean.shape
    assert y_var.shape == f_var.shape
    assert np.all(y_var != f_var)  # The mean might not change but the covariance should

    # 2. Run tests with gp_layer outputting f_sample

    gp_layer.returns_samples = True

    f_sample = gp_layer(X)
    y_sample = likelihood_layer(f_sample)  # training flag does not matter here

    assert y_sample.shape == f_sample.shape
    # note: currently we don't draw a sample of y_sample the likelihood!
    # this should be changed!
    assert np.all(y_sample == f_sample.numpy())


@pytest.mark.parametrize("GPFlowLikelihood", TEST_GPFLOW_LIKELIHOODS)
def test_add_losses(GPFlowLikelihood):
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)
    likelihood_layer = LikelihoodLayer(GPFlowLikelihood())

    # 1. Run tests with gp layer outputting f_mean, f_var
    gp_layer.returns_samples = False

    f_mean, f_var = gp_layer(X)
    y_mean, y_var = likelihood_layer((f_mean, f_var), training=True, targets=Y)

    expected_loss = -np.sum(
        np.mean(likelihood_layer.variational_expectations(f_mean, f_var, Y), axis=0)
    )
    np.testing.assert_almost_equal(likelihood_layer.losses, [expected_loss], decimal=11)
    assert likelihood_layer.losses != 0

    y_mean, y_var = likelihood_layer((f_mean, f_var), training=False)
    assert likelihood_layer.losses == [0.0]

    # 2. Run tests with gp_layer outputting f_sample
    gp_layer.returns_samples = True

    f_sample = gp_layer(X)
    _ = likelihood_layer(f_sample, training=True, targets=Y)
    # TODO: test for y_sample return value ...

    expected_loss = -np.sum(np.mean(likelihood_layer.log_prob(f_sample, Y), axis=0))
    np.testing.assert_almost_equal(likelihood_layer.losses, [expected_loss], decimal=11)
    assert likelihood_layer.losses != 0

    y_mean, y_var = likelihood_layer((f_mean, f_var), training=False)
    assert likelihood_layer.losses == [0.0]


@pytest.mark.parametrize("GPflowLikelihood", TEST_GPFLOW_LIKELIHOODS)
def test_gpflow_consistency(GPflowLikelihood):
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)
    likelihood_layer = LikelihoodLayer(GPflowLikelihood())
    gp_layer.build(X.shape)

    f_samples, f_mean, f_var = gp_layer.predict(X, num_samples=5)

    layer_varexp = likelihood_layer.variational_expectations(f_mean, f_var, Y)
    gpflow_varexp = GPflowLikelihood().variational_expectations(f_mean, f_var, Y)
    assert np.all(layer_varexp == gpflow_varexp)

    layer_mean, layer_var = likelihood_layer.predict_mean_and_var(f_mean, f_var)
    gpflow_mean, gpflow_var = GPflowLikelihood().predict_mean_and_var(f_mean, f_var)
    assert np.all(layer_mean == gpflow_mean)
    assert np.all(layer_var == gpflow_var)

    layer_log_prob = likelihood_layer.log_prob(f_samples, Y)
    gpflow_log_prob = GPflowLikelihood().log_prob(f_samples, Y)
    assert np.all(layer_log_prob == gpflow_log_prob)
