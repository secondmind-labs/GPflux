#
# Copyright (c) 2021 The GPflux Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
from typing import Union

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow import Parameter
from gpflow.models.model import RegressionData
from gpflow.utilities import positive, to_default_float

import gpflux

tf.keras.backend.set_floatx("float64")


class LogPrior_ELBO_SVGP(gpflow.models.SVGP):
    """
    SVGP model that takes into account the log_prior in the ELBO
    """

    def elbo(self, data: RegressionData) -> tf.Tensor:
        loss_prior = tf.add_n([p.log_prior_density() for p in self.trainable_parameters])
        return super().elbo(data) + loss_prior


def load_data():
    path = os.path.join(os.path.dirname(__file__), "..", "snelson1d.npz")
    data = np.load(path)
    return (data["X"], data["Y"])


def get_num_data(data):
    X, Y = data
    assert len(X) == len(Y)
    return len(X)


def make_dataset(data, as_dict=True):
    X, Y = data
    dataset_base = {"inputs": X, "targets": Y} if as_dict else (X, Y)
    batch_size = get_num_data(data)
    return tf.data.Dataset.from_tensor_slices(dataset_base).batch(batch_size)


def make_kernel_likelihood_iv():
    kernel = gpflow.kernels.SquaredExponential(variance=0.7, lengthscales=0.6)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        to_default_float(1.0), to_default_float(0.5)
    )
    likelihood = gpflow.likelihoods.Gaussian(variance=0.08)
    Z = np.linspace(0, 6, 20)[:, np.newaxis]
    inducing_variable = gpflow.inducing_variables.InducingPoints(Z)
    gpflow.set_trainable(inducing_variable, False)
    return kernel, likelihood, inducing_variable


def create_gpflow_svgp(kernel, likelihood, inducing_variable):
    return LogPrior_ELBO_SVGP(kernel, likelihood, inducing_variable)


def create_gp_layer(kernel, inducing_variable, num_data):
    mok = gpflow.kernels.SharedIndependent(kernel, output_dim=1)
    moiv = gpflow.inducing_variables.SharedIndependentInducingVariables(inducing_variable)
    return gpflux.layers.GPLayer(mok, moiv, num_data, mean_function=gpflow.mean_functions.Zero())


def create_gpflux_sldgp(kernel, likelihood, inducing_variable, num_data):
    gp_layer = create_gp_layer(kernel, inducing_variable, num_data)
    likelihood_layer = gpflux.layers.LikelihoodLayer(likelihood)
    model = gpflux.models.DeepGP([gp_layer], likelihood_layer, num_data=num_data)
    return model


def create_gpflux_sequential_and_loss(kernel, likelihood, inducing_variable, num_data):
    gp_layer = create_gp_layer(kernel, inducing_variable, num_data)
    loss = gpflux.losses.LikelihoodLoss(likelihood)
    likelihood_container = gpflux.layers.TrackableLayer()
    likelihood_container.likelihood = likelihood  # for likelihood to be discovered as trainable
    model = tf.keras.Sequential([gp_layer, likelihood_container])
    return model, loss


def assign_svgp_to_sldgp(svgp, sldgp):
    [gp] = sldgp.f_layers
    gp.q_mu.assign(svgp.q_mu)
    gp.q_sqrt.assign(svgp.q_sqrt)
    kernel = gp.kernel.kernel
    kernel.variance.assign(svgp.kernel.variance)
    kernel.lengthscales.assign(svgp.kernel.lengthscales)
    iv = gp.inducing_variable.inducing_variable
    iv.Z.assign(svgp.inducing_variable.Z)
    sldgp.likelihood_layer.likelihood.variance.assign(svgp.likelihood.variance)


def fit_scipy(model, data, maxiter=100):
    def training_loss():
        return -model.elbo(data)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(training_loss, model.trainable_variables, options=dict(maxiter=maxiter))


def assert_equivalence(svgp, sldgp, data, **tol_kws):
    X, Y = data
    np.testing.assert_allclose(sldgp.elbo(data), svgp.elbo(data), **tol_kws)
    np.testing.assert_allclose(sldgp.predict_f(X), svgp.predict_f(X), **tol_kws)


def test_svgp_equivalence_after_assign():
    data = load_data()
    svgp = create_gpflow_svgp(*make_kernel_likelihood_iv())
    fit_scipy(svgp, data)
    sldgp = create_gpflux_sldgp(*make_kernel_likelihood_iv(), get_num_data(data))
    assign_svgp_to_sldgp(svgp, sldgp)
    assert_equivalence(svgp, sldgp, data)


def fit_adam(
    model: Union[gpflow.models.SVGP, gpflux.models.DeepGP], data, maxiter, adam_learning_rate=0.01
):
    X, Y = data
    num_data = len(X)

    def training_loss():
        """
        NOTE: the Keras model.compile()/fit() uses the implicit losses, which are computed as

        >>> _ = model(data, training=True)
        >>> return tf.reduce_sum(model.losses)

        The scaling factor leads to a O(1e-3) discrepancy between approaches; to have an exact
        comparison we therefore re-scale the objective here.
        """
        return -model.elbo(data) / num_data

    adam = tf.optimizers.Adam(adam_learning_rate)

    @tf.function
    def optimization_step():
        adam.minimize(training_loss, var_list=model.trainable_variables)

    for i in range(maxiter):
        optimization_step()


def _keras_fit_adam(model, dataset, maxiter, adam_learning_rate=0.01, loss=None):
    model.compile(optimizer=tf.optimizers.Adam(adam_learning_rate), loss=loss)
    model.fit(dataset, epochs=maxiter)


def keras_fit_adam(sldgp: gpflux.models.DeepGP, data, maxiter, adam_learning_rate=0.01):
    model = sldgp.as_training_model()
    dataset = make_dataset(data)
    _keras_fit_adam(model, dataset, maxiter, adam_learning_rate=adam_learning_rate)


def _keras_fit_natgrad(
    base_model,
    dataset,
    maxiter,
    adam_learning_rate=0.01,
    gamma=1.0,
    loss=None,
    run_eagerly=None,
):
    model = gpflux.optimization.NatGradWrapper(base_model)
    model.natgrad_layers = True  # Shortcut to apply natural gradients to all layers
    natgrad = gpflow.optimizers.NaturalGradient(gamma=gamma)
    adam = tf.optimizers.Adam(adam_learning_rate)
    model.compile(
        optimizer=[natgrad, adam],
        loss=loss,
        run_eagerly=run_eagerly,
    )
    model.fit(dataset, epochs=maxiter)


def keras_fit_natgrad(
    sldgp,
    data,
    maxiter,
    adam_learning_rate=0.01,
    gamma=1.0,
    run_eagerly=None,
):
    base_model = sldgp.as_training_model()
    dataset = make_dataset(data)
    _keras_fit_natgrad(
        base_model,
        dataset,
        maxiter,
        adam_learning_rate=adam_learning_rate,
        gamma=gamma,
        run_eagerly=run_eagerly,
    )


def fit_natgrad(model, data, maxiter, adam_learning_rate=0.01, gamma=1.0):
    if isinstance(model, gpflow.models.SVGP):
        variational_params = [(model.q_mu, model.q_sqrt)]
    else:
        [layer] = model.f_layers
        variational_params = [(layer.q_mu, layer.q_sqrt)]

    variational_params_vars = []
    for param_list in variational_params:
        these_vars = []
        for param in param_list:
            gpflow.set_trainable(param, False)
            these_vars.append(param.unconstrained_variable)
        variational_params_vars.append(these_vars)
    hyperparam_variables = model.trainable_variables

    num_data = get_num_data(data)

    @tf.function
    def training_loss():
        return -model.elbo(data) / num_data

    natgrad = gpflow.optimizers.NaturalGradient(gamma=gamma)
    adam = tf.optimizers.Adam(adam_learning_rate)

    @tf.function
    def optimization_step():
        """
        NOTE: In GPflow, we would normally do alternating ascent:

        >>> natgrad.minimize(training_loss, var_list=variational_params)
        >>> adam.minimize(training_loss, var_list=hyperparam_variables)

        This, however, does not match up with the single pass we require for Keras's
        model.compile()/fit(). Hence we manually re-create the same optimization step.
        """
        with tf.GradientTape() as tape:
            tape.watch(variational_params_vars)
            loss = training_loss()
        variational_grads, other_grads = tape.gradient(
            loss, (variational_params_vars, hyperparam_variables)
        )
        for (q_mu_grad, q_sqrt_grad), (q_mu, q_sqrt) in zip(variational_grads, variational_params):
            natgrad._natgrad_apply_gradients(q_mu_grad, q_sqrt_grad, q_mu, q_sqrt)
        adam.apply_gradients(zip(other_grads, hyperparam_variables))

    for i in range(maxiter):
        optimization_step()


@pytest.mark.parametrize(
    "svgp_fitter, sldgp_fitter",
    [
        (fit_adam, fit_adam),
        (fit_adam, keras_fit_adam),
        (fit_natgrad, fit_natgrad),
        (fit_natgrad, keras_fit_natgrad),
    ],
)
def test_svgp_equivalence_with_sldgp(svgp_fitter, sldgp_fitter, maxiter=20):
    data = load_data()

    svgp = create_gpflow_svgp(*make_kernel_likelihood_iv())
    svgp_fitter(svgp, data, maxiter=maxiter)

    sldgp = create_gpflux_sldgp(*make_kernel_likelihood_iv(), get_num_data(data))
    sldgp_fitter(sldgp, data, maxiter=maxiter)

    assert_equivalence(svgp, sldgp, data)


@pytest.mark.parametrize(
    "svgp_fitter, keras_fitter, tol_kw",
    [
        (fit_adam, _keras_fit_adam, {}),
        (fit_natgrad, _keras_fit_natgrad, dict(atol=1e-8)),
    ],
)
def test_svgp_equivalence_with_keras_sequential(svgp_fitter, keras_fitter, tol_kw, maxiter=10):
    X, Y = data = load_data()

    svgp = create_gpflow_svgp(*make_kernel_likelihood_iv())
    svgp_fitter(svgp, data, maxiter=maxiter)

    keras_model, loss = create_gpflux_sequential_and_loss(
        *make_kernel_likelihood_iv(), get_num_data(data)
    )
    keras_fitter(keras_model, make_dataset(data, as_dict=False), maxiter, loss=loss)

    f_dist = keras_model(X)
    np.testing.assert_allclose((f_dist.loc, f_dist.scale.diag ** 2), svgp.predict_f(X), **tol_kw)


def run_gpflux_sldgp(data, optimizer, maxiter):
    kernel, likelihood, inducing_variable = make_kernel_likelihood_iv()
    num_data = len(data[0])
    model = create_gpflux_sldgp(kernel, likelihood, inducing_variable, num_data)
    if optimizer == "natgrad":
        fit_natgrad(model, data, maxiter=maxiter)
    elif optimizer == "adam":
        fit_adam(model, data, maxiter=maxiter)
    elif optimizer == "scipy":
        pytest.skip("Numerically unstable")
        fit_scipy(model, data, maxiter=maxiter)
    elif optimizer == "keras_adam":
        keras_fit_adam(model, data, maxiter=maxiter)
    elif optimizer == "keras_natgrad":
        keras_fit_natgrad(
            model, data, maxiter=maxiter, run_eagerly=True
        )  # run_eagerly needed so codecov can pick up the lines
    else:
        raise NotImplementedError
    return model


def run_gpflow_svgp(data, optimizer, maxiter):
    kernel, likelihood, inducing_variable = make_kernel_likelihood_iv()
    model = create_gpflow_svgp(kernel, likelihood, inducing_variable)
    if optimizer == "natgrad":
        fit_natgrad(model, data, maxiter=maxiter)
    elif optimizer == "adam":
        fit_adam(model, data, maxiter=maxiter)
    elif optimizer == "scipy":
        fit_scipy(model, data, maxiter=maxiter)
    else:
        raise NotImplementedError
    return model


@pytest.mark.parametrize(
    "optimizer",
    ["natgrad", "adam", "scipy", "keras_adam", "keras_natgrad"],
)
def test_run_gpflux_sldgp(optimizer):
    data = load_data()
    _ = run_gpflux_sldgp(data, optimizer, maxiter=10)


@pytest.mark.parametrize("optimizer", ["natgrad", "adam", "scipy"])
def test_run_gpflow_svgp(optimizer):
    data = load_data()
    _ = run_gpflow_svgp(data, optimizer, maxiter=10)
