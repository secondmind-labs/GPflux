# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Deep GP modelling: 1D

In this notebook, we demonstrate how to construct and fit a deep GP model to a toy dataset.
"""
# %%
import numpy as np
import tensorflow as tf
import gpflow
import gpflux
from gpflow.ci_utils import ci_niter

import matplotlib.pyplot as plt

# %%
from gpflux.layers import LikelihoodLoss

tf.keras.backend.set_floatx("float64")

# %%
# %matplotlib inline

# %%
num_data = 20
noisy_input = noisy_output = False
if noisy_input:
    X = np.random.rand(num_data, 1)
else:
    X = np.linspace(0, 1, num_data).reshape(-1, 1)
Y = (X >= 0.5).astype(float)
if noisy_output:
    Y += np.random.randn(num_data, 1) * 0.05

# %%
input_dim = output_dim = 1

# %%
plt.figure()
plt.scatter(X, Y)
plt.show()

# %%
num_inducing = 13
Zinit = np.linspace(-0.1, 1.1, num_inducing)[:, None]


def create_model_svgp():
    kernel = gpflow.kernels.SquaredExponential()
    likelihood = gpflow.likelihoods.Gaussian(0.01)
    inducing_variable = gpflow.inducing_variables.InducingPoints(Zinit)
    gpflow.set_trainable(inducing_variable, False)
    model = gpflow.models.SVGP(kernel, likelihood, inducing_variable)
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss_closure((X, Y)), model.trainable_variables)
    return model


# %%
model = create_model_svgp()

# %%
Xtest = np.linspace(-0.5, 1.5, 100)[:, None]

# %%
Fmu, Fvar = model.predict_f(Xtest)


# %%
def plot_gp(X, Fmu, Fvar):
    X = X.squeeze()
    Fmu = Fmu.numpy().squeeze()
    Fvar = Fvar.numpy().squeeze()
    plt.plot(X, Fmu)
    lower = Fmu - 2 * np.sqrt(Fvar)
    upper = Fmu + 2 * np.sqrt(Fvar)
    plt.fill_between(X, lower, upper, alpha=0.1)


# %%
plot_gp(Xtest, Fmu, Fvar)
plt.scatter(X, Y)


# %%
def create_model_one_layer(model_class=tf.keras.Model):
    # num_inducing = 13
    # init_first_layer = gpflux.initializers.KmeansInitializer(X, num_inducing)
    init_first_layer = gpflux.initializers.GivenZInitializer(Zinit)

    gp_layer = gpflux.helpers.construct_gp_layer(
        num_data, num_inducing, input_dim, output_dim, initializer=init_first_layer
    )
    gpflow.set_trainable(gp_layer.inducing_variable, False)

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.01))

    inputs = tf.keras.Input((input_dim,))
    f = gp_layer(inputs)
    outputs = likelihood_layer(f)

    return model_class(inputs=inputs, outputs=outputs)


# %%
def create_model_two_layers(model_class=tf.keras.Model):
    hidden_dim = 1

    # num_inducing = 13
    # init_first_layer = gpflux.initializers.KmeansInitializer(X, num_inducing)
    init_first_layer = gpflux.initializers.GivenZInitializer(Zinit)

    layer1 = gpflux.helpers.construct_gp_layer(
        num_data, num_inducing, input_dim, hidden_dim, initializer=init_first_layer
    )
    layer1.mean_function = gpflow.mean_functions.Identity()  # TODO: pass layer_type instead
    layer1.q_sqrt.assign(layer1.q_sqrt * 0.01)
    layer1._num_samples = 1000
    gpflow.set_trainable(layer1.inducing_variable, False)

    # FeedForwardInitializer does not handle multi samples
    # init_last_layer = gpflux.initializers.FeedForwardInitializer()
    init_last_layer = gpflux.initializers.GivenZInitializer(Zinit)

    layer2 = gpflux.helpers.construct_gp_layer(
        num_data, num_inducing, hidden_dim, output_dim, initializer=init_last_layer
    )
    gpflow.set_trainable(layer2.inducing_variable, False)

    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.01))

    inputs = tf.keras.Input((input_dim,))
    f1 = layer1(inputs)
    f2 = layer2(f1)
    outputs = likelihood_layer(f2)

    return model_class(inputs=inputs, outputs=outputs)


# %%
batch_size = 20
num_epochs = ci_niter(1000)


# %%
def do_fit(model, lr=0.1):
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", patience=5, factor=0.95, verbose=1, min_lr=1e-6,
        )
    ]

    opt = tf.optimizers.Adam(learning_rate=lr)
    loss = LikelihoodLoss(model.layers[-1].likelihood)
    model.compile(opt, loss=loss)

    history = model.fit(x=X, y=Y, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks)
    return history


# %%
sldgp = create_model_one_layer(tf.keras.Model)
history_1 = do_fit(sldgp)

# %%
dgp = create_model_two_layers(tf.keras.Model)
history_2 = do_fit(dgp, lr=0.01)

# %%
res = sldgp(Xtest)
res2 = dgp(Xtest)

# %%
import tensorflow_probability as tfp


# %%
def make_mixture(out):
    num_samples = out.shape[0]
    # equal_prob = tf.convert_to_tensor(
    #     np.full([out.shape[1], num_samples], 1/num_samples),
    #     tf.float64
    # )
    # cat = tfp.distributions.Categorical(probs=equal_prob)
    # components = [
    #     tfp.distributions.MultivariateNormalDiag(loc=Fmu, scale_diag=tf.sqrt(Fvar))
    #     for Fmu, Fvar in zip(out.f_mu, out.f_var)
    # ]
    # return tfp.distributions.Mixture(cat, components)
    equal_prob = tf.convert_to_tensor(np.full([num_samples], 1 / num_samples), tf.float64)
    cat = tfp.distributions.Categorical(probs=equal_prob)
    Fmu = tf.transpose(out.f_mu, [1, 0, 2])
    Fvar = tf.transpose(out.f_var, [1, 0, 2])
    components = tfp.distributions.MultivariateNormalDiag(loc=Fmu, scale_diag=tf.sqrt(Fvar))
    return tfp.distributions.MixtureSameFamily(cat, components)


# %%
def plot_dgp(X, out):
    d = make_mixture(out)
    Fmu = d.mean()
    Fvar = d.variance()
    plot_gp(X, Fmu, Fvar)


# %%
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plot_gp(Xtest, Fmu, Fvar)
plt.scatter(X, Y)

plt.subplot(1, 3, 2)
plot_gp(Xtest, res.f_mu, res.f_var)
plt.scatter(X, Y)

plt.subplot(1, 3, 3)
plot_dgp(Xtest, res2)
plt.scatter(X, Y)

# %%
