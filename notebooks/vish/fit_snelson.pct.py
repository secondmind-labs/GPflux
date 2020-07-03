import os
import numpy as np
import matplotlib.pyplot as plt

from bayesian_benchmarks import data as uci_datasets

import gpflow
from gpflux.vish.inducing_variables import SphericalHarmonicInducingVariable
from gpflux.vish.kernels import Matern, Parameterised, ArcCosine
from gpflux.vish.spherical_harmonics import SphericalHarmonicsCollection
from gpflux.vish.helpers import preprocess_data

from notebooks.vish.ci_utils import is_running_pytest


def get_snelson():
    path = os.path.join(".", "snelson1d.npz")
    data = np.load(path)
    return data["X"], data["Y"]


def get_uci(dataset):
    data_class = getattr(uci_datasets, dataset)
    data = data_class(0, prop=0.9)
    return data
    # data_train = data.X_train, data.Y_train
    # data_test = data.X_test, data.Y_test
    # return data_train, data_test


# +
MAXITER = 3 if is_running_pytest() else 1000
MODEL_TYPE = "vish"
MAX_DEGREE = 25
TRUNCATION_LEVEL = 25
KERNEL_TYPE = "ArcCosine"

if KERNEL_TYPE == "Matern":
    NU = 0.5
# -


data, _, _ = preprocess_data(get_snelson())
data_train = data
data_test = data

# data = get_uci("Yacht")
# from vish.helpers import add_bias
# data_train = add_bias(data.X_train), data.Y_train
# data_test = add_bias(data.X_test), data.Y_test


def build_model(dimension):

    if KERNEL_TYPE == "Matern":
        kernel = Matern(
            dimension,
            truncation_level=TRUNCATION_LEVEL,
            nu=NU,
            # weight_variances=np.ones(dimension)
        )
        degrees = range(MAX_DEGREE)
    elif KERNEL_TYPE == "Param":
        kernel = Parameterised(dimension, truncation_level=TRUNCATION_LEVEL,)
        degrees = range(MAX_DEGREE)
    elif KERNEL_TYPE == "ArcCosine":
        kernel = ArcCosine(dimension, truncation_level=MAX_DEGREE, weight_variances=np.ones(dimension))
        degrees = kernel.degrees

    print(degrees)
    harmonics = SphericalHarmonicsCollection(dimension, degrees=degrees)
    inducing_variable = SphericalHarmonicInducingVariable(harmonics)

    return gpflow.models.SVGP(
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=inducing_variable,
        whiten=False,
    )


def build_model_svgp(dimension):
    from vish.helpers import get_num_inducing

    if KERNEL_TYPE == "ArcCosine":
        kernel = gpflow.kernels.ArcCosine(order=1, weight_variances=np.ones(dimension))
        num_inducing = get_num_inducing("arccosine", dimension, MAX_DEGREE)
        print(num_inducing)
        del kernel.bias_variance
        kernel.bias_variance = 0
    # gpflow.set_trainable(kernel.bias_variance, False)
    Z = np.random.randn(num_inducing, dimension)
    inducing_variable = gpflow.inducing_variables.InducingPoints(Z)

    return gpflow.models.SVGP(
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=inducing_variable,
        whiten=False,
    )


if MODEL_TYPE == "vish":
    model = build_model(data_train[0].shape[1])
elif MODEL_TYPE == "svgp":
    model = build_model_svgp(data_train[0].shape[1])

elbo1 = model.elbo(data_train)

# gpflow.set_trainable(model.kernel.variance, False)

opt = gpflow.optimizers.Scipy()
try:
    opt.minimize(
        model.training_loss_closure(data_train),
        model.trainable_variables,
        options=dict(maxiter=MAXITER, disp=True),
    )
except KeyboardInterrupt:
    print("Ctrl-c")
elbo2 = model.elbo(data_train)
print("before:", elbo1.numpy())
print("after:", elbo2.numpy())
model

# +
plt.figure()
X_data = data[0][:, :1]
Y_data = data[1]
N_test = 500
X_new = np.linspace(X_data.min() - 0.5, X_data.max() + 0.5, N_test).reshape(-1, 1)  # [N_test, 1]
X_new = np.c_[X_new, np.ones((N_test, 2))]  # [N_test, 3]
f_mean, f_var = model.predict_y(X_new)

import matplotlib.pyplot as plt
plt.ylim(-5, 5)
plt.plot(X_data, Y_data, "o")
X_new = X_new[:, :1]
plt.plot(X_new, f_mean, "C1")
plt.plot(X_new, f_mean + np.sqrt(f_var + 1e-6), "C1--")
plt.plot(X_new, f_mean - np.sqrt(f_var + 1e-6), "C1--")
plt.show()
# -

gpflow.utilities.print_summary(model)

# Evaluate
import pprint
from scipy.stats import norm


XT, YT = data_test
mu, var = model.predict_y(XT)

d = YT - mu
l = norm.logpdf(YT, loc=mu, scale=var ** 0.5)
mse = np.average(d ** 2)
rmse = mse ** 0.5
nlpd = -np.average(l)

res = dict(rmse=rmse, mse=mse, nlpd=nlpd)
pprint.pprint(res)


import matplotlib.pyplot as plt
from vish.conditional import Lambda_diag_elements
Lambda = Lambda_diag_elements(model.inducing_variable, model.kernel).numpy()
plt.figure()
plt.plot(Lambda, "o")
plt.plot(model.q_mu.numpy(), "o")

plt.figure()
plt.matshow(model.q_sqrt[0].numpy())
plt.show()