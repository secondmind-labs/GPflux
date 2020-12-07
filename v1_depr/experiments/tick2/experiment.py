import datetime
import os
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
from model_builder_utils import build_model, cluster_patches, save_model_parameters
from sacred import Experiment
from sklearn import preprocessing
from utils import (
    calc_binary_error,
    calc_multiclass_error,
    get_dataset,
    get_error_cb,
    save_gpflow_model,
)

import gpflow
import gpflow.training.monitor as mon

import gpflux

NAME = "mnist"
ex = Experiment(NAME)

tf.logging.set_verbosity(tf.logging.FATAL)
warnings.filterwarnings('ignore')


@ex.config
def config():
    model_type = "deep-conv"  # convgp | cnn

    dataset = "mnist27"

    lr_cfg = {
        "type": "exp",
        "steps": int(100e3),
        "rate": 0.25,
        "start": 0.01
    }

    date = datetime.datetime.now().strftime('%d-%H-%M')

    iterations = int(300e3)
    patch_shape = [5, 5]
    batch_size = 32
    # path to save results
    basepath = "./"

    with_indexing = False
    with_weights = False
    num_layers = 1

    num_inducing_points = 384
    base_kern = "RBF"
    init_patches = "kmeans"  # 'patches', 'random'
    restore = False
    init_file = None
    cout = 10
    feature_maps_out = [cout] * (num_layers - 1)
    like = "bern"

    # print hz
    hz = {
        'slow': 10000,
        'short': 50
    }


@ex.capture
def get_data(dataset, model_type):
    (X, Y), (Xs, Ys) = get_dataset(dataset)

    if "mnist" in dataset:
        H, W = 28, 28
        X = X.reshape(-1, H, W, 1)
    elif dataset == "cifar":
        H, W = 32, 32
        X = X.reshape(-1, H, W, 3)

    # Xs = Xs.reshape(-1, H, W, 1)
    def print_array_info(name, a):
        print(f'{name}.shape: {a.shape}| {name}.min: {a.min()}| {name}.max: {a.max()}')

    print_array_info('X', X)
    print_array_info('Y', Y)
    print_array_info('Xs', Xs)
    print_array_info('Ys', Ys)
    return (X, Y), (Xs, Ys)


@ex.capture
def experiment_name(model_type, lr_cfg, num_inducing_points, batch_size, dataset,
                    base_kern, init_file, patch_shape, with_weights,
                    with_indexing, date, num_layers, cout, like):
    name = f"{model_type}_{date}"
    args = np.array([
            name,
            dataset,
            f"W-{with_weights}",
            f"Ti-{with_indexing}",
            f"init-{init_file is not None}",
            f"{like}",
            f"kern-{base_kern}",
            f"lr-{lr_cfg['type']}",
            f"-{lr_cfg['start']}",
            f"-{lr_cfg['steps']}",
            f"-{lr_cfg['rate']}",
            f"M-{num_inducing_points}",
            f"N-{batch_size}",
            f"Cout-{cout}",
            f"L-{num_layers}",
            f"patch-{patch_shape[0]}"])
    return "_".join(args.astype(str))

@ex.capture
def experiment_name_short(model_type, date, dataset, with_weights, with_indexing, num_layers, like):
    name = f"{model_type}_{date}"
    args = np.array([
            name,
            dataset,
            f"W-{with_weights}",
            f"Ti-{with_indexing}",
            f"{like}",
            f"L-{num_layers}"
    ])
    return "_".join(args.astype(str))

@ex.capture
def experiment_path(basepath, dataset):
    experiment_dir = Path(basepath, dataset, experiment_name())
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return str(experiment_dir)


#########
## ConvGP
#########


@ex.capture
def restore_session(session, restore, basepath):
    model_path = experiment_path()
    if restore and os.path.isdir(model_path):
        mon.restore_session(session, model_path)
        print("Model restored")



@ex.capture
def convgp_setup_model(train_data, batch_size,
                       patch_shape, num_inducing_points,
                       feature_maps_out, with_weights,
                       with_indexing, init_file, like):

    X, Y = train_data
    num_layers = len(feature_maps_out) + 1

    model = build_model(
        X,
        Y,
        num_layers=num_layers,
        feature_maps_out=feature_maps_out,
        patch_shape=patch_shape,
        num_inducing_points=num_inducing_points,
        tick=with_indexing,
        weights=with_weights,
        init_file=init_file,
        likelihood=like,
        batch_size=batch_size
    )

    return model


@ex.capture
def convgp_monitor_tasks(train_data, model, optimizer, hz, basepath, dataset):
    Xs, Ys = train_data
    path = experiment_path()
    fw = mon.LogdirWriter(path)

    tasks = []

    def lr(*args, **kwargs):
        sess = model.enquire_session()
        return sess.run(optimizer._optimizer._lr)

    def save_params(*args, **kwargs):
        filename = experiment_path() + f'/params'
        save_model_parameters(model, filename)

    def periodic_short():
        return mon.PeriodicIterationCondition(hz['short'])

    def periodic_slow():
        return mon.PeriodicIterationCondition(hz['slow'])

    tasks += [
        mon.ScalarFuncToTensorBoardTask(fw, lr, "lr")
            .with_name('lr')
            .with_condition(periodic_short())
            .with_exit_condition(True)
            .with_flush_immediately(True)]

    tasks += [
        mon.ScalarFuncToTensorBoardTask(fw, save_params, "save")
            .with_name('save')
            .with_condition(periodic_slow())
            .with_exit_condition(True)
            .with_flush_immediately(True)]

    tasks += [
        mon.ModelToTensorBoardTask(fw, model)
            .with_name('model_tboard')
            .with_condition(periodic_short())
            .with_exit_condition(True)
            .with_flush_immediately(True)]

    tasks += [
        mon.PrintTimingsTask().with_name('print')
            .with_condition(periodic_short())
            .with_exit_condition(True)]

    if dataset == "mnist01" or dataset == "mnist27":
        print("Binary error")
        error_func = calc_binary_error
    else:
        print("multiclass error")
        error_func = calc_multiclass_error

    f1 = get_error_cb(model, Xs, Ys, error_func, full=False)
    tasks += [
        mon.ScalarFuncToTensorBoardTask(fw, f1, "error")
            .with_name('error_test')
            .with_condition(periodic_short())
            .with_exit_condition(True)
            .with_flush_immediately(True)]

    f2 = get_error_cb(model, Xs, Ys, error_func, full=True)
    tasks += [
        mon.ScalarFuncToTensorBoardTask(fw, f2, "error_full")
            .with_name('error_full_test')
            .with_condition(periodic_slow())
            .with_exit_condition(True)
            .with_flush_immediately(True)]

    return tasks


@ex.capture
def convgp_setup_optimizer(model, global_step, iterations, lr_cfg):
    if lr_cfg['type'] == "custom":
        denom = 1 + tf.cast(global_step // lr_cfg['steps'], gpflow.settings.float_type)
        lr = lr_cfg['start'] / denom

    elif lr_cfg['type'] == "exp":
        lr = tf.train.exponential_decay(
            lr_cfg["start"],
            global_step,
            lr_cfg["steps"],
            lr_cfg["rate"],
            staircase=True,
        )
    elif lr_cfg['type'] == "expaim":
        total_steps = iterations
        t = global_step
        stair_length = lr_cfg['stairlen']
        lr_init = lr_cfg['start']
        scale = lr_cfg.get('scale', 0.1)
        rate = lr_cfg['rate']
        t = tf.cast(t - (t + 1) % stair_length, tf.float64)
        lr = lr_init * (scale ** (t / (rate * total_steps)))
    elif lr_cfg['type'] == "tanh":
        T = tf.cast(iterations, tf.float64)
        ti = global_step
        t = tf.cast(global_step, tf.float64)

        def lr_param(name, dtype=tf.float64): return tf.cast(lr_cfg[name], dtype)
        stair_length = lr_param('stairlen', tf.int32)
        lr_min = lr_param('lr_min')
        lr_max = lr_param('lr_max')
        upper = lr_param('upper')
        lower = lr_param('lower')

        t = tf.cast(ti - (ti + 1) % stair_length, tf.float64)
        tanh = tf.tanh(lower * (1.0 - t / T) + upper * (t / T))
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 - tanh)
    else:
        lr = lr_cfg['lr']
    return gpflow.train.AdamOptimizer(lr)


@ex.capture
def convgp_fit(train_data, test_data, iterations):
    session = gpflow.get_default_session()
    step = mon.create_global_step(session)
    model = convgp_setup_model(train_data)
    # model.compile()
    print(model)
    optimizer = convgp_setup_optimizer(model, step)
    monitor_tasks = convgp_monitor_tasks(test_data, model, optimizer)
    monitor = mon.Monitor(monitor_tasks, session, step, print_summary=True)
    try:
        with monitor:
            optimizer.minimize(model,
                            step_callback=monitor,
                            maxiter=iterations,
                            global_step=step)
    finally:
        filename = experiment_path() + f'/params'
        save_model_parameters(model, filename)
        convgp_finish(train_data, test_data, model)


def convgp_save(model):
    filename = experiment_path() + f'/convgp.gpflow'
    save_gpflow_model(filename, model)
    print(f"Model saved at {filename}")


@ex.capture
def convgp_finish(train_data, test_data, model, dataset):
    X, Y = train_data
    Xs, Ys = test_data
    error_func = calc_binary_error if dataset in ["mnist01", "mnist27"] else calc_multiclass_error
    error_func = get_error_cb(model, Xs, Ys, error_func, full=True)
    print(f"Error test: {error_func()}")
    convgp_save(model)


@ex.command
def calc_error():
   from utils import calc_multiclass_error
   train_data, test_data = get_data()
   Xs, Ys = test_data
   model = convgp_setup_model(train_data)
   print(model)
   error = calc_multiclass_error(model, Xs, Ys, batchsize=32, mc=5)
   print("error", error)

@ex.command
def plot_patches():
    from plotting import plot_patches
    train_data, test_data = get_data()
    model = convgp_setup_model(train_data)
    print(model)
    plot_patches(model)

@ex.command
def plot_g2():
    train_data, test_data = get_data()
    X, Y = train_data
    X = X.reshape(-1, 28**2)

    model = convgp_setup_model(train_data)
    flag = np.load("hits_tick_misses_conv.npy")
    print(flag.shape)

    model.layers[0].kern.__class__ = gpflux.convolution.ConvKernel
    m, v, s = model.plot_g(X[flag])

    vals = dict(
        X=X[flag],
        m=m,
        v=v,
        s=s
    )
    np.save(experiment_name_short() + "__hits_misses.npy", vals)

@ex.command
def plot_g():
    train_data, test_data = get_data()
    X, Y = train_data
    Xs, Ys = test_data
    Ns = 100

    model = convgp_setup_model(train_data)

    from utils import get_miclassified_binary

    # hits = get_miclassified_binary(model, Xs, Ys)
    # hits = hits.flatten()
    X = X.reshape(-1, 28**2)
    hits_train = get_miclassified_binary(model, X, Y)
    hits = hits_train.flatten()
    print(hits.shape)
    print(hits.sum())
    print((~hits).sum())

    model.layers[0].kern.__class__ = gpflux.convolution.ConvKernel
    m_correct, v_correct, s_correct = model.plot_g(X[hits][:Ns])
    m_wrong, v_wrong, s_wrong = model.plot_g(X[~hits])

    vals = dict(
        X_correct=X[hits][:Ns],
        Y_correct=Y[hits][:Ns],
        m_correct=m_correct,
        v_correct=v_correct,
        s_correct=s_correct,
        X_wrong=X[~hits],
        Y_wrong=Y[~hits],
        m_wrong=m_wrong,
        v_wrong=v_wrong,
        s_wrong=s_wrong,
        hits=hits
    )

    np.save(experiment_name_short() + "__train_plotting.npy", vals)

    # from plotting import plot_g_mean_var_samples
    # import matplotlib.pyplot as plt

    # fig, axes = plt.subplots(2 * Ns, 1 + 1 + 1 + len(s))
    # plot_g_mean_var_samples(Xs[:Ns], m, v, s, fig, axes[:Ns, :], titles=True)

@ex.capture
@ex.automain
def main(model_type):
    train_data, test_data = get_data()
    convgp_fit(train_data, test_data)
