import datetime
import json
import pprint
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np
import tensorflow as tf
from bayesian_benchmarks import data as uci_datasets
from bayesian_benchmarks.data import Dataset
from sacred import Experiment
from scipy.stats import norm

from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from utils import ExperimentName, git_version

tf.keras.backend.set_floatx("float64")

THIS_DIR = Path(__file__).parent
LOGS = THIS_DIR / "tmp"
EXPERIMENT = Experiment("UCI")


@EXPERIMENT.config
def config():
    # Timestamp (None)
    date = datetime.datetime.now().strftime("%b%d_%H%M%S")
    # Dataset (None)
    dataset = "Yacht"
    # Dataset split (None)
    split = 0
    # Number of layers (Ignore)
    num_layers = 1
    # Model name (None)
    model_name = f"dgp-{num_layers}"
    # number of inducing points (Ignore)
    num_inducing = 256
    # batch size (Ignore)
    batch_size = 1024
    # Number of times the complete training dataset is iterated over (Ignore)
    num_epochs = 1000


@EXPERIMENT.capture
def experiment_name(_config: Any) -> Optional[str]:
    config = _config.copy()
    del config["seed"]
    return ExperimentName(EXPERIMENT, config).get()


def get_dataset_class(dataset) -> Type[Dataset]:
    return getattr(uci_datasets, dataset)


@EXPERIMENT.capture
def get_data(split, dataset):
    data = get_dataset_class(dataset)(split=split)
    return data


@EXPERIMENT.capture
def build_model(X, num_inducing, num_layers):
    config = Config(
        num_inducing=num_inducing,
        inner_layer_qsqrt_factor=1e-5,
        between_layer_noise_variance=1e-2,
        likelihood_noise_variance=1e-2,
        white=True
    )
    model = build_constant_input_dim_deep_gp(
        X,
        num_layers,
        config=config,
    )
    return model


@EXPERIMENT.capture
def train_model(model: tf.keras.models.Model, data_train, batch_size, num_epochs):
    X_train, Y_train = data_train
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            'loss', factor=0.95, patience=3, min_lr=1e-6, verbose=1
        ),
    ]
    model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=1,
    )


def evaluate_model(model, data_test):
    XT, YT = data_test
    out = model(XT)
    y_mean, y_var = out.y_mean, out.y_var
    d = YT - y_mean
    l = norm.logpdf(YT, loc=y_mean, scale=y_var ** 0.5)
    mse = np.average(d ** 2)
    rmse = mse ** 0.5
    nlpd = -np.average(l)
    return dict(rmse=rmse, mse=mse, nlpd=nlpd)


@EXPERIMENT.automain
def main(_config):
    data = get_data()
    model = build_model(data.X_train)

    model.compile(optimizer=tf.optimizers.Adam(0.01))
    train_model(model, (data.X_train, data.Y_train))

    metrics = evaluate_model(model, (data.X_test, data.Y_test))

    _dict = {**_config, **metrics, "commit": git_version()}
    with open(f"{LOGS}/{experiment_name()}.json", "w") as fp:
        json.dump(_dict, fp, indent=2)

    print("=" * 60)
    print(experiment_name())
    pprint.pprint(_dict)
    print("=" * 60)
