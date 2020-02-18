import datetime
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sacred import Experiment

import gpflow
from model_builder_utils import build_model
from utils import calc_multiclass_error, compute_predictions, get_dataset


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.FATAL)
warnings.filterwarnings('ignore')

analyse = Experiment("analyse")


@analyse.config
def config():
    ##################
    # Plot parameters
    eventfiles = None
    paramfiles = None
    x_axis = "relative"  # "step"

    ##################
    # Model parameters
    dataset = "mnist"
    models_configs = {
        "mnist_convgp_layer_1": dict(num_layers=1, init_file="~/experiments/tick2/mnist/deep-conv_18-15-12_mnist_W-True_Ti-False_init-False_white-True_soft_kern-RBF_lr-tanh_-0.01_-100000_-0.25_M-384_N-32_Cout-10_L-1_patch-5/params.npy"),
        "mnist_convgp_layer_2": dict(num_layers=2, init_file="~/experiments/tick2/mnist/deep-conv_16-15-35_mnist_W-True_Ti-False_init-False_white-True_soft_kern-RBF_lr-tanh_-0.01_-100000_-0.25_M-384_N-32_Cout-10_L-2_patch-5/params.npy"),
        "mnist_convgp_layer_3": dict(num_layers=3, init_file="~/experiments/tick2/mnist/deep-conv_17-13-41_mnist_W-True_Ti-False_init-True_white-True_soft_kern-RBF_lr-tanh_-0.01_-100000_-0.25_M-384_N-32_Cout-10_L-3_patch-5/params.npy"),
        "mnist_tick_layer_1": dict(num_layers=1, with_indexing=True, init_file="~/experiments/tick2/mnist/deep-conv_18-15-11_mnist_W-True_Ti-True_init-False_white-True_soft_kern-RBF_lr-tanh_-0.01_-100000_-0.25_M-384_N-32_Cout-10_L-1_patch-5/params.npy"),
        "mnist_tick_layer_2": dict(num_layers=2, with_indexing=True, init_file="~/experiments/tick2/mnist/deep-conv_16-15-30_mnist_W-True_Ti-True_init-False_white-True_soft_kern-RBF_lr-tanh_-0.01_-100000_-0.25_M-384_N-32_Cout-10_L-2_patch-5/params.npy"),
        "mnist_tick_layer_3": dict(num_layers=3, with_indexing=True, init_file="~/experiments/tick2/mnist/deep-conv_17-13-39_mnist_W-True_Ti-True_init-True_white-True_soft_kern-RBF_lr-tanh_-0.01_-100000_-0.25_M-384_N-32_Cout-10_L-3_patch-5/params.npy"),
    }

    # dataset = "cifar"
    # model_config = {
    #     "cifar_convgp_layer_2": dict(num_layers=2, init_file="~/experiments/tick2/cifar/deep-conv_16-23-55_cifar_W-True_Ti-False_init-False_white-True_soft_kern-RBF_lr-tanh_-0.01_-100000_-0.25_M-384_N-32_Cout-10_L-2_patch-5/params.npy"),
    #     "cifar_convgp_layer_3": dict(num_layers=3, init_file="~/experiments/tick2/cifar/deep-conv_17-15-31_cifar_W-True_Ti-False_init-True_white-True_soft_kern-RBF_lr-tanh_-0.01_-100000_-0.25_M-384_N-32_Cout-10_L-3_patch-5/params.npy"),
    #     "cifar_tick_layer_2": dict(num_layers=2, with_indexing=True, init_file="~/experiments/tick2/cifar/deep-conv_16-23-53_cifar_W-True_Ti-True_init-False_white-True_soft_kern-RBF_lr-tanh_-0.01_-100000_-0.25_M-384_N-32_Cout-10_L-2_patch-5/params.npy"),
    #     "cifar_tick_layer_3": dict(num_layers=3, with_indexing=True, init_file="~/experiments/tick2/cifar/deep-conv_17-15-30_cifar_W-True_Ti-True_init-True_white-True_soft_kern-RBF_lr-tanh_-0.01_-100000_-0.25_M-384_N-32_Cout-10_L-3_patch-5/params.npy"),
    # }



@analyse.capture
def get_data(dataset):
    (X, Y), (Xs, Ys) = get_dataset(dataset)

    if "mnist" in dataset:
        H, W = 28, 28
        X = X.reshape(-1, H, W, 1)
    elif dataset == "cifar":
        H, W = 32, 32
        X = X.reshape(-1, H, W, 3)

    return (X, Y), (Xs, Ys)


@analyse.command
def plot_error_full(eventfiles, x_axis):
    """
    Extracts `error_full` tagged data from TensorBoard
    event files and plots them using matplotlib.
    """
    eventfiles = eventfiles if isinstance(eventfiles, list) else [eventfiles]
    events = []
    color = None
    for i, event_file in enumerate(eventfiles):
        event = error_full_values(event_file)
        event['relative'] = (event.timestamp - event.timestamp.iloc[0]) / (60 ** 2)   # Convert to hours
        x = event.relative if x_axis == 'relative' else event.index
        linestyle = '--'
        if i % 2 == 0:
            linestyle = '-'
            color = None
        line = plt.plot(x, event.value, linestyle=linestyle, color=color)[0]
        color = line.get_color()
    plt.show()


@analyse.command
def error(models_configs):
    """
    """
    train, (Xs, Ys) = get_data()
    for name, model_config in models_configs.items():
        gpflow.reset_default_graph_and_session()
        model = setup_model(train, model_config)
        error = calc_multiclass_error(model, Xs, Ys, batchsize=32, mc=5)
        print(f"Error {name}: {error}")


@analyse.command
def nlpp(models_configs):
    """
    NLPP of models
    """
    train_data, test_data = get_data()
    _, true_labels = test_data
    for name, model_config in models_configs.items():
        gpflow.reset_default_graph_and_session()
        model = setup_model(train_data, model_config)
        preds = compute_predictions(model, test_data)

        def _nlpp(probs):
            jitter = 1e-12
            return -np.mean(np.log(probs + jitter))

        full_nlpp = _nlpp(label_probs(preds['probs'], true_labels))
        miss_nlpp = _nlpp(label_probs(preds['probs_miss'], preds['ys_miss_true']))

        print(f"NLPP '{name}': full={full_nlpp}, miss={miss_nlpp}")


def label_probs(probs, labels):
    return np.array([probs[i, label] for i, label in enumerate(labels)])


def mesh_with_default_model_config(cfg):
    new_cfg = dict(
        patch_shape=[5, 5],
        batch_size=32,
        num_inducing_points=384,
        base_kern="RBF",
        cout=10,
        white=True,
        like="soft",
        with_indexing=False,
        with_weights=True)

    new_cfg.update(cfg)
    cout = new_cfg['cout']
    num_layers = new_cfg['num_layers']
    new_cfg['feature_maps_out'] = [cout] * (num_layers - 1)
    return new_cfg



def setup_model(train_data, config):
    X, Y = train_data
    cfg = mesh_with_default_model_config(config)
    init_file_path = Path(cfg['init_file']).expanduser()
    assert init_file_path.exists(), f"Init file '{init_file_path}' not found"
    init_file = str(init_file_path)
    model = build_model(
        X,
        Y,
        num_layers=cfg['num_layers'],
        feature_maps_out=cfg['feature_maps_out'],
        patch_shape=cfg['patch_shape'],
        num_inducing_points=cfg['num_inducing_points'],
        tick=cfg['with_indexing'],
        weights=cfg['with_weights'],
        init_file=init_file,
        white=cfg['white'],
        likelihood=cfg['like'],
        batch_size=cfg['batch_size']
    )
    return model


def error_full_values(event_file: str):
    assert Path(event_file).exists(), f"Cannot find event file {event_file}"
    event_file = str(Path(event_file).expanduser())
    df = pd.DataFrame([])
    for event in tf.train.summary_iterator(event_file):
        for v in event.summary.value:
            if v.tag == 'error_full':
                time = event.wall_time
                value = event.summary.value[0].simple_value
                df = df.append(pd.DataFrame([[time, value]], index=[event.step], columns=['timestamp', 'value']))
    return df


@analyse.automain
def main():
    return 0
