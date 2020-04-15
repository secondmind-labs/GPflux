import os
import pytest
import numpy as np
import tensorflow as tf
import gpflow
import gpflux

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
    moiv = gpflow.inducing_variables.SharedIndependentInducingVariables(
        inducing_variable
    )
    gp_layer = gpflux.layers.GPLayer(
        mok,
        moiv,
        num_data,
        initializer=gpflux.initializers.ZZeroOneInitializer(Z=inducing_variable.Z),
        mean_function=gpflow.mean_functions.Zero(),
        returns_samples=False,
    )
    likelihood_layer = gpflux.layers.LikelihoodLayer(likelihood)
    model = gpflux.models.DeepGP([gp_layer], likelihood_layer)
    return model


def assign_svgp_to_sldgp(svgp, sldgp):
    _ = sldgp.predict_f(
        svgp.inducing_variable.Z
    )  # make sure keras layers have built; need to call as .build() does not work

    gp = sldgp.gp_layers[0]
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
    opt.minimize(
        training_loss, model.trainable_variables, options=dict(maxiter=maxiter)
    )


def assert_equivalence(svgp, sldgp, data, **tol_kws):
    X, Y = data
    np.testing.assert_allclose(svgp.elbo(data), sldgp.elbo(data), **tol_kws)
    np.testing.assert_allclose(svgp.predict_f(X), sldgp.predict_f(X), **tol_kws)


def test_svgp_equivalence_after_assign():
    X, Y = data = load_data()
    svgp = create_gpflow_svgp(*make_kernel_likelihood_iv())
    fit_scipy(svgp, data)
    sldgp = create_gpflux_sldgp(*make_kernel_likelihood_iv(), get_num_data(data))
    assign_svgp_to_sldgp(svgp, sldgp)
    assert_equivalence(svgp, sldgp, data)


def fit_adam(model, data, maxiter, adam_learning_rate=0.01):
    def training_loss():
        return -model.elbo(data)

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


@pytest.mark.parametrize(
    "sldgp_fitter, tol_kws",
    [
        (fit_adam, {}),
        (keras_fit_adam, dict(rtol=1e-3, atol=1e-5)),
    ],  # NOTE smallest tolerance that passes
)
def test_svgp_equivalence_with_adam(sldgp_fitter, tol_kws, maxiter=500):
    X, Y = data = load_data()
    svgp = create_gpflow_svgp(*make_kernel_likelihood_iv())
    fit_adam(svgp, data, maxiter=maxiter)
    sldgp = create_gpflux_sldgp(*make_kernel_likelihood_iv(), get_num_data(data))
    sldgp_fitter(sldgp, data, maxiter=maxiter)
    assert_equivalence(svgp, sldgp, data, **tol_kws)
