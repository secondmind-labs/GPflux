import os

import numpy as np
import pytest
import tensorflow as tf

import gpflow

import gpflux

from gpflux.layers import LikelihoodLoss

tf.keras.backend.set_floatx("float64")


def load_data():
    path = os.path.join(os.path.dirname(__file__), "..", "snelson1d.npz")
    data = np.load(path)
    return (data["X"], data["Y"])


def get_num_data(data):
    X, Y = data
    assert len(X) == len(Y)
    return len(X)


def make_kernel_likelihood_iv():
    kernel = gpflow.kernels.SquaredExponential(variance=0.7, lengthscales=0.6)
    likelihood = gpflow.likelihoods.Gaussian(variance=0.08)
    Z = np.linspace(0, 6, 20)[:, np.newaxis]
    inducing_variable = gpflow.inducing_variables.InducingPoints(Z)
    gpflow.set_trainable(inducing_variable, False)
    return kernel, likelihood, inducing_variable


def create_gpflow_svgp(kernel, likelihood, inducing_variable):
    return gpflow.models.SVGP(kernel, likelihood, inducing_variable)


def create_gpflux_sldgp(kernel, likelihood, inducing_variable, num_data):
    mok = gpflow.kernels.SharedIndependent(kernel, output_dim=1)
    moiv = gpflow.inducing_variables.SharedIndependentInducingVariables(inducing_variable)
    gp_layer = gpflux.layers.GPLayer(
        mok,
        moiv,
        num_data,
        initializer=gpflux.initializers.GivenZInitializer(Z=inducing_variable.Z),
        mean_function=gpflow.mean_functions.Zero(),
    )
    likelihood_layer = gpflux.layers.LikelihoodLayer(likelihood)
    model = gpflux.models.DeepGP([gp_layer], likelihood_layer)
    # The model must be compiled to add the loss function.
    model.compile()
    return model


def assign_svgp_to_sldgp(svgp, sldgp):
    # NOTE: We need to call the prediction method first to make sure that the
    # Keras layers have built and the initializers have been called already;
    # otherwise the initializers would overwrite the assigned q_mu/q_sqrt!
    _ = sldgp.predict_f(
        svgp.inducing_variable.Z
    )  # We cannot call .build() directly as the inputs to the model are (X, Y) tuple, not a float

    gp = sldgp.gp_layers[0]
    gp.q_mu.assign(svgp.q_mu)
    gp.q_sqrt.assign(svgp.q_sqrt)
    kernel = gp.kernel.kernel
    kernel.variance.assign(svgp.kernel.variance)
    kernel.lengthscales.assign(svgp.kernel.lengthscales)
    iv = gp.inducing_variable.inducing_variable
    iv.Z.assign(svgp.inducing_variable.Z)
    sldgp.likelihood.variance.assign(svgp.likelihood.variance)


def fit_scipy(model, data, maxiter=100):
    def training_loss():
        return -model.elbo(data)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(training_loss, model.trainable_variables, options=dict(maxiter=maxiter))


def assert_equivalence(svgp, sldgp, data, **tol_kws):
    X, Y = data
    np.testing.assert_allclose(svgp.elbo(data), sldgp.elbo(data), **tol_kws)
    f_dist = sldgp.predict_f(X)
    np.testing.assert_allclose(svgp.predict_f(X), (f_dist.mean(), f_dist.scale.diag), **tol_kws)


def test_svgp_equivalence_after_assign():
    data = load_data()
    svgp = create_gpflow_svgp(*make_kernel_likelihood_iv())
    fit_scipy(svgp, data)
    sldgp = create_gpflux_sldgp(*make_kernel_likelihood_iv(), get_num_data(data))
    assign_svgp_to_sldgp(svgp, sldgp)
    assert_equivalence(svgp, sldgp, data)


def fit_adam(model, data, maxiter, adam_learning_rate=0.01):
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


def keras_fit_adam(model, data, maxiter, adam_learning_rate=0.01):
    adam = tf.optimizers.Adam(adam_learning_rate)
    model.compile(optimizer=adam)
    X, Y = data
    dataset_tuple = ((X, Y), Y)
    batch_size = len(X)
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset_tuple).batch(batch_size)
    model.fit(train_dataset, epochs=maxiter)


@pytest.mark.parametrize("sldgp_fitter", [fit_adam, keras_fit_adam])
def test_svgp_equivalence_with_adam(sldgp_fitter, maxiter=500):
    X, Y = data = load_data()

    svgp = create_gpflow_svgp(*make_kernel_likelihood_iv())
    fit_adam(svgp, data, maxiter=maxiter)

    sldgp = create_gpflux_sldgp(*make_kernel_likelihood_iv(), get_num_data(data))
    sldgp_fitter(sldgp, data, maxiter=maxiter)

    assert_equivalence(svgp, sldgp, data)


def keras_fit_natgrad(
    base_model,
    data,
    gamma=1.0,
    adam_learning_rate=0.01,
    maxiter=1000,
    use_other_loss_fn=False,
    run_eagerly=None,
):
    if use_other_loss_fn:

        def other_loss_fn():
            return -base_model.elbo(data)

    else:
        other_loss_fn = None

    model = gpflux.optimization.NatGradWrapper(base_model, other_loss_fn=other_loss_fn)
    natgrad = gpflow.optimizers.NaturalGradient(gamma=gamma)
    adam = tf.optimizers.Adam(adam_learning_rate)
    likelihood = base_model.likelihood
    model.compile(
        optimizer=[natgrad, adam], loss=LikelihoodLoss(likelihood), run_eagerly=run_eagerly,
    )
    X, Y = data
    dataset_tuple = (X, Y)
    batch_size = len(X)
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset_tuple).batch(batch_size)
    model.fit(train_dataset, epochs=maxiter)


def fit_natgrad(model, data, gamma=1.0, adam_learning_rate=0.01, maxiter=1000):
    if isinstance(model, gpflow.models.SVGP):
        variational_params = [(model.q_mu, model.q_sqrt)]
    else:
        layer = model.gp_layers[0]
        variational_params = [(layer.q_mu, layer.q_sqrt)]

    variational_params_vars = []
    for param_list in variational_params:
        these_vars = []
        for param in param_list:
            gpflow.set_trainable(param, False)
            these_vars.append(param.unconstrained_variable)
        variational_params_vars.append(these_vars)
    hyperparam_variables = model.trainable_variables

    X, Y = data
    num_data = len(X)

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
    "sldgp_fitter", [fit_natgrad, keras_fit_natgrad],
)
def test_svgp_equivalence_with_natgrad(sldgp_fitter):
    maxiter = 10
    X, Y = data = load_data()

    svgp = create_gpflow_svgp(*make_kernel_likelihood_iv())
    fit_natgrad(svgp, data, maxiter=maxiter, gamma=1.0)

    sldgp = create_gpflux_sldgp(*make_kernel_likelihood_iv(), get_num_data(data))
    sldgp_fitter(sldgp, data, maxiter=maxiter, gamma=1.0)

    # Absolute tolerance was reduced as part of PR#139
    assert_equivalence(svgp, sldgp, data, atol=1e-8)


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
    "optimizer", ["natgrad", "adam", "scipy", "keras_adam", "keras_natgrad"],
)
def test_run_gpflux_sldgp(optimizer):
    data = load_data()
    _ = run_gpflux_sldgp(data, optimizer, maxiter=10)


@pytest.mark.parametrize("optimizer", ["natgrad", "adam", "scipy"])
def test_run_gpflow_svgp(optimizer):
    data = load_data()
    _ = run_gpflow_svgp(data, optimizer, maxiter=10)
