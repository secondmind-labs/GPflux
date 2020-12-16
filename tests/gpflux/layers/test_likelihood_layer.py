import tensorflow as tf
import numpy as np
import pytest

from gpflow.kernels import Matern52
from gpflow.likelihoods import Bernoulli, Beta, Gaussian, Poisson
from gpflow.mean_functions import Zero

from gpflux.initializers import GivenZInitializer
from gpflux.helpers import construct_basic_kernel, construct_basic_inducing_variables
from gpflux.layers import GPLayer, LikelihoodLayer
from gpflux.layers.likelihood_layer import LikelihoodLoss, LikelihoodOutputs

TEST_GPFLOW_LIKELIHOODS = [Bernoulli, Beta, Gaussian, Poisson]


def setup_gp_layer_and_data(num_inducing: int, **gp_layer_kwargs):
    input_dim = 30
    output_dim = 5
    num_data = 100
    data = make_data(input_dim, output_dim, num_data=num_data)

    kernel = construct_basic_kernel(Matern52(), output_dim)
    inducing_vars = construct_basic_inducing_variables(num_inducing, input_dim, output_dim)
    initializer = GivenZInitializer()
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
    Y = [np.random.multivariate_normal(np.zeros(num_data), cov)[:, None] for _ in range(output_dim)]
    Y = np.hstack(Y)
    return X, Y


@pytest.mark.parametrize("GPFlowLikelihood", TEST_GPFLOW_LIKELIHOODS)
def test_call_shapes(GPFlowLikelihood):
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)
    likelihood_layer = LikelihoodLayer(GPFlowLikelihood(), returns_samples=False)

    # 1. Run tests with gp layer outputting f_mean, f_var
    f_distribution = gp_layer(X)
    y_dist_params = likelihood_layer(f_distribution)

    assert y_dist_params.y_mean.shape == f_distribution.shape
    assert y_dist_params.y_var.shape == f_distribution.scale.diag.shape
    # The mean might not change but the covariance should
    assert np.all(y_dist_params.y_var != f_distribution.covariance())

    # 2. Run tests with likelihood outputting f_sample and y_sample
    likelihood_layer.returns_samples = True

    f_distribution = gp_layer(X)
    # y_sample not yet defined
    f_sample, _ = likelihood_layer(f_distribution)  # training flag does not matter here

    assert f_sample.shape == f_distribution.shape
    # note: currently we don't draw a sample of y_sample the likelihood!
    # this should be changed!
    assert np.all(f_sample == tf.convert_to_tensor(f_distribution).numpy())


@pytest.mark.parametrize("GPFlowLikelihood", TEST_GPFLOW_LIKELIHOODS)
def test_losses(GPFlowLikelihood):
    gp_layer, (X, Y) = setup_gp_layer_and_data(num_inducing=5)
    likelihood = GPFlowLikelihood()
    likelihood_layer = LikelihoodLayer(likelihood, returns_samples=False)
    likelihood_loss = LikelihoodLoss(likelihood)

    # 1. Run tests with gp layer outputting f_mean, f_var
    f_distribution = gp_layer(X)
    y_distribution_params = likelihood_layer(f_distribution)
    f_mean = f_distribution.loc
    f_var = f_distribution.scale.diag

    expected_loss = -np.sum(np.mean(likelihood.variational_expectations(f_mean, f_var, Y), axis=0))
    np.testing.assert_almost_equal(
        likelihood_loss(Y, y_distribution_params), expected_loss, decimal=5
    )

    # 2. Run tests with gp_layer outputting f_sample
    likelihood_layer.returns_samples = True

    f_sample = gp_layer(X)
    # y_samples is not yet implemented. Assume that y_samples = f_samples
    f_samples, y_samples = likelihood_layer(f_sample)

    expected_loss = -np.sum(np.mean(likelihood.log_prob(f_sample, Y), axis=0))
    np.testing.assert_almost_equal(
        likelihood_loss(Y, (f_samples, y_samples)), expected_loss, decimal=5
    )


def test_tensor_coercible():
    f_mu = tf.zeros([1, 2])
    f_var = tf.zeros([1, 2])
    y_mu = tf.ones([1, 2])
    y_var = tf.zeros([1, 2])
    tensor_coercible = LikelihoodOutputs(f_mu, f_var, y_mu, y_var)

    np.testing.assert_array_equal(y_mu, tf.convert_to_tensor(tensor_coercible))
