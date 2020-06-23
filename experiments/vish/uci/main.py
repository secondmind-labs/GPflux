import datetime
import math
import json
import os
import pprint
import sys
import time
from dataclasses import dataclass
from typing import Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sacred import Experiment
from sacred.observers import FileStorageObserver
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score

import gpflux
import gpflow
from bayesian_benchmarks import data as uci_datasets
from bayesian_benchmarks.data import Dataset
from vish.helpers import (
    add_bias,
    get_max_degree_closest_but_smaller_than_num_inducing,
)
from vish.inducing_variables import SphericalHarmonicInducingVariable
from vish.kernels import ArcCosine, Matern, Parameterised
from vish.spherical_harmonics import SphericalHarmonicsCollection

from utils import ExperimentName

tf.keras.backend.set_floatx("float64")

# LOGS = "logs/tmp"
LOGS = f"logs/{datetime.datetime.now().strftime('%b%d')}/Wilson"
ex = Experiment("uci")
ex.observers.append(FileStorageObserver(f"{LOGS}/runs"))


@ex.config
def config():
    model_type = "vish"
    kernel_type = "arc"
    date = datetime.datetime.now().strftime("%b%d_%H%M%S")
    # dataset needs to correspond to the exact name in bayesian_benchmarks.data
    # e.g. Power, Wilson_protein, Wilson_3droad, etc.
    dataset = "Yacht"
    # number of inducing points
    max_num_inducing = 1024
    (
        max_degree,
        num_inducing,
    ) = get_max_degree_closest_but_smaller_than_num_inducing(
        kernel_type="matern" if "mat" in kernel_type else "arccosine",
        # dimension + 1 to account for the bias
        dimension=get_dataset_class(dataset).D + 1,
        num_inducing=max_num_inducing,
    )
    epochs = 200
    batch_size = 1024
    split = 0
    prop = 0.9
    natgrad = False
    framework = "gpflow"
    truncation_level = max_degree


def get_dataset_class(dataset) -> Type[Dataset]:
    return getattr(uci_datasets, dataset)


def get_path():
    return f"./{LOGS}/{experiment_name()}"


@ex.capture
def get_data(split, dataset, prop):
    data = get_dataset_class(dataset)(split=split, prop=prop)
    print("DATASET N_TRAIN", len(data.X_train))
    print("DATASET N_TEST", len(data.X_test))
    return data


@ex.capture
def experiment_name(
    dataset,
    date,
    model_type,
    max_degree,
    kernel_type,
    split,
    natgrad,
    max_num_inducing,
    num_inducing,
    truncation_level,
    framework,
):
    return (
        ExperimentName(date)
        .add("model", model_type)
        .add("dataset", dataset)
        .add("split", split)
        .add("maxdegree", max_degree)
        .add("kernel", kernel_type)
        .add("natgrad", natgrad)
        .add("Mmax", max_num_inducing)
        .add("M", num_inducing)
        .add("L", truncation_level)
        .add("framework", framework)
        .get()
    )


@ex.capture
def experiment_info_dict(
    dataset,
    date,
    model_type,
    max_degree,
    kernel_type,
    split,
    natgrad,
    prop,
    max_num_inducing,
    num_inducing,
    truncation_level,
):
    return dict(
        model=model_type,
        dataset=dataset,
        kernel=kernel_type,
        split=split,
        natgrad=natgrad,
        prop=prop,
        M_degree=max_degree,
        M_max=max_num_inducing,
        M=num_inducing,
        L=truncation_level,
    )


@ex.capture
def build_kernel_and_inducing_variable(
    dimension,
    kernel_type,
    max_degree,
    model_type,
    num_inducing,
    truncation_level,
):
    if model_type == "vish":
        if "mat" in kernel_type:
            if "12" in kernel_type:
                nu = 0.5
            elif "32" in kernel_type:
                nu = 1.5
            elif "52" in kernel_type:
                nu = 2.5
            else:
                raise NotImplementedError

            kernel = Matern(
                dimension,
                truncation_level=truncation_level,
                nu=nu,
                weight_variances=np.ones(dimension),
            )
            degrees = kernel.degrees[:max_degree]
        elif "arc" in kernel_type:
            kernel = ArcCosine(
                dimension,
                weight_variances=np.ones(dimension),
                truncation_level=max_degree,
            )
            degrees = kernel.degrees
        else:
            raise NotImplementedError

        harmonics = SphericalHarmonicsCollection(dimension, degrees=degrees)
        inducing_variable = SphericalHarmonicInducingVariable(harmonics)
        return kernel, inducing_variable
    elif model_type == "svgp":
        if "mat" in kernel_type:
            kernel = gpflow.kernels.Matern32(lengthscales=np.ones(dimension))
        elif "arc" in kernel_type:
            kernel = gpflow.kernels.ArcCosine(
                order=1, weight_variances=np.ones(dimension)
            )
        Z = np.random.randn(num_inducing, dimension)
        inducing_variable = gpflow.inducing_variables.InducingPoints(Z)
        return kernel, inducing_variable


@ex.capture
def build_model(data_train, model_type, framework):
    if framework == "gpflow":
        model = build_gpflow_model(data_train)
        layer = model
    elif framework == "gpflux":
        model = build_gpflux_model(data_train)
        layer = model.layers[1]
    else:
        raise NotImplementedError

    if model_type == "vish":
        print("Init")
        from vish.conditional import Lambda_diag_elements
        q_sqrt_init = np.diag(
            Lambda_diag_elements(layer.inducing_variable, layer.kernel).numpy() ** 0.5
        )  # [M, M]
        q_mu_init = q_sqrt_init @ np.random.randn(len(q_sqrt_init), 1)  # [M, 1]
        layer.q_sqrt.assign(q_sqrt_init[None])
        layer.q_mu.assign(q_mu_init)

    return model


@ex.capture
def build_gpflux_model(data_train, kernel_type, max_degree, natgrad):
    X, Y = data_train
    num_data, dimension = X.shape
    kernel, inducing_variable = build_kernel_and_inducing_variable(dimension)

    gp_layer = gpflux.layers.GPLayer(
        kernel=kernel,
        inducing_variable=inducing_variable,
        mean_function=gpflow.mean_functions.Zero(),
        num_data=num_data,
        num_latent_gps=Y.shape[1],
        returns_samples=False,
        verify=False,
        white=False,
    )
    gp_layer._initialized = True

    likelihood_layer = gpflux.layers.LikelihoodLayer(
        gpflow.likelihoods.Gaussian(1.0)
    )

    inputs = tf.keras.Input((dimension,), name="inputs")
    targets = tf.keras.Input((1,), name="targets")

    f1 = gp_layer(inputs)
    outputs = likelihood_layer(f1, targets=targets)
    model = tf.keras.Model(inputs=(inputs, targets), outputs=(outputs))
    adam = tf.keras.optimizers.Adam(learning_rate=1e-2)

    if natgrad:
        model = gpflux.optimization.NatGradWrapper(model)
        # gamma = 0.9 beta1 = 0.7  # default: 0.9 beta2 = 0.9  # default: 0.99
        natgrad = gpflux.optimization.MomentumNaturalGradient()
        optimizer = [natgrad, adam]
    else:
        optimizer = adam

    model.compile(optimizer=optimizer)
    return model


@ex.capture
def build_gpflow_model(data_train, model_type, kernel_type, max_degree, natgrad):
    X, Y = data_train
    num_data, dimension = X.shape
    kernel, inducing_variable = build_kernel_and_inducing_variable(dimension)
    model = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=inducing_variable,
        whiten=False,
        num_data=num_data,
    )
    return model


@ex.capture
def train_model(model, data_train, epochs, batch_size):
    if isinstance(model, gpflow.models.BayesianModel):
        return train_gpflow_model(model, data_train)
    else:
        return train_gpflux_model(model, data_train)


@ex.capture
def train_gpflow_model(model, data_train, epochs, batch_size):
    X, Y = data_train
    num_data = len(X)

    if batch_size == -1:
        batch_size = num_data

    iter_per_epoch = math.ceil(num_data / batch_size)
    maxiter = iter_per_epoch * epochs
    print("Maxiter", maxiter)

    # try:
    time_start = time.time()
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        model.training_loss_closure(data_train),
        model.trainable_variables,
        options=dict(maxiter=maxiter, disp=1),
    )
    # except KeyboardInterrupt:
    #     print("Training stopped")
    # finally:
    time_end = time.time()
    return time_end - time_start


@ex.capture
def train_gpflux_model(model, data_train, epochs, batch_size):
    if batch_size == -1:
        batch_size = len(data_train[0])

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", patience=5, factor=0.95, verbose=1, min_lr=1e-6,
        ),
        gpflux.callbacks.TensorBoard(get_path()),
    ]
    try:
        time_start = time.time()
        history = model.fit(
            x=data_train,
            y=None,
            batch_size=min(batch_size, len(data_train[0])),
            epochs=epochs,
            callbacks=callbacks,
        )
        # plt.plot(history.history["loss"])
    except KeyboardInterrupt:
        print("Training stopped")
    finally:
        time_end = time.time()

    gpflow.utilities.print_summary(model)
    return time_end - time_start


def evaluate_model(model, data_test):
    XT, YT = data_test

    if isinstance(model, gpflow.models.BayesianModel):
        mu, var = model.predict_y(XT)
    else:
        data_test_Y_nan = (XT, YT * np.nan)
        mu, var = model.predict(data_test_Y_nan, verbose=1, batch_size=1000)

    d = YT - mu
    l = norm.logpdf(YT, loc=mu, scale=var ** 0.5)
    mse = np.average(d ** 2)
    rmse = mse ** 0.5
    nlpd = -np.average(l)

    return dict(rmse=rmse, mse=mse, nlpd=nlpd)


@ex.automain
def main(dataset, split):
    experiment_name()

    data = get_data(split)
    data_train = add_bias(data.X_train), data.Y_train
    data_test = add_bias(data.X_test), data.Y_test

    # Model
    model = build_model(data_train)
    gpflow.utilities.print_summary(model)

    duration = train_model(model, data_train)

    # Evaluation
    experiment_metrics = evaluate_model(model, data_test)
    # merge two dictionaries
    experiment_dict = {**experiment_info_dict(), **experiment_metrics}
    experiment_dict["time"] = duration

    with open(f"{get_path()}_results.json", "w") as fp:
        json.dump(experiment_dict, fp)

    import pprint

    print(experiment_name())
    pprint.pprint(experiment_dict)
