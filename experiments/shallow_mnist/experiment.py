import datetime
import os
from pathlib import Path

import numpy as np
from sacred import Experiment
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.initializers import constant, truncated_normal
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, MaxPool2D)
from tensorflow.keras.regularizers import l2

import gpflow
import gpflow.training.monitor as mon
import gpflux
from utils import (calc_binary_error, calc_multiclass_error, get_dataset,
                   get_error_cb, save_gpflow_model)

NAME = "mnist"
ex = Experiment(NAME)


@ex.config
def config():
    model_type = "convgp"  # convgp | cnn

    dataset = "mnist"

    lr_cfg = {
        "decay": "custom",
        "lr": 1e-4
    }

    date = datetime.datetime.now().strftime('%d-%H-%M')

    iterations = 50000
    patch_shape = [5, 5]
    batch_size = 128
    # path to save results
    basepath = "./"

    num_inducing_points = 1000
    base_kern = "RBF"
    with_weights = True
    with_indexing = True
    init_patches = "patches-unique"  # 'patches', 'random'
    restore = False

    # print hz
    hz = {
        'slow': 1000,
        'short': 50
    }


@ex.capture
def get_data(dataset, model_type):
    (X, Y), (Xs, Ys) = get_dataset(dataset)
    if model_type == "cnn":
        if dataset == "mnist":
            H, W = 28, 28
        elif dataset == "cifar":
            H, W = 32, 32
        X = X.reshape(-1, H, W, 1)
        Xs = Xs.reshape(-1, H, W, 1)
    return (X, Y), (Xs, Ys)


@ex.capture
def experiment_name(model_type, lr_cfg, num_inducing_points, batch_size, dataset,
                    base_kern, init_patches, patch_shape,
                    with_weights, with_indexing, date):
    name = f"{model_type}_{date}"
    if model_type == "cnn":
        args = np.array([
                name,
                f"lr-{lr_cfg['lr']}",
                f"batchsize-{batch_size}"])
    else:
        args = np.array([
                name,
                f"W-{with_weights}",
                f"Ti-{with_indexing}",
                f"initpatches-{init_patches}",
                f"kern-{base_kern}",
                f"lr-{lr_cfg['lr']}",
                f"lrdecay-{lr_cfg['decay']}",
                f"nip-{num_inducing_points}",
                f"batchsize-{batch_size}",
                f"patch-{patch_shape[0]}"])
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
def get_likelihood(dataset):
    if dataset == "mnist01":
        return gpflow.likelihoods.Bernoulli()
    return gpflow.likelihoods.SoftMax(10)


@ex.capture
def patch_initializer(X, H, W, init_patches):
    if init_patches == "random":
        return gpflux.init.NormalInitializer()
    unique = init_patches == "patches-unique"
    return gpflux.init.PatchSamplerInitializer(X, width=W, height=H, unique=unique)


@gpflow.defer_build()
@ex.capture
def convgp_setup_model(train_data, batch_size,
                       patch_shape, num_inducing_points,
                       with_weights, with_indexing):

    X, Y = train_data
    H = int(X.shape[1] ** .5)

    likelihood = get_likelihood()
    num_latents = likelihood.num_classes if hasattr(likelihood, 'num_classes') else 1

    patches = patch_initializer(X[:100], H, H)

    layer0 = gpflux.layers.WeightedSumConvLayer(
        [H, H],
        num_inducing_points,
        patch_shape,
        num_latents=num_latents,
        with_indexing=with_indexing,
        with_weights=with_weights,
        patches_initializer=patches)

    layer0.kern.basekern.variance = 25.0
    layer0.kern.basekern.lengthscales = 1.2

    # init kernel
    if with_indexing:
        layer0.kern.index_kernel.variance = 25.0
        layer0.kern.index_kernel.lengthscales = 3.0

    # break symmetry in variational parameters
    layer0.q_sqrt = layer0.q_sqrt.read_value()
    layer0.q_mu = np.random.randn(*(layer0.q_mu.read_value().shape))

    model = gpflux.DeepGP(X, Y,
                          layers=[layer0],
                          likelihood=likelihood,
                          batch_size=batch_size,
                          name="my_deep_gp")
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
        mon.CheckpointTask(path)
            .with_name('saver')
            .with_condition(periodic_short())]

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

    error_func = calc_binary_error if dataset == "mnist01" \
        else calc_multiclass_error

    f1 = get_error_cb(model, Xs, Ys, error_func)
    tasks += [
        mon.ScalarFuncToTensorBoardTask(fw, f1, "error")
            .with_name('error')
            .with_condition(periodic_short())
            .with_exit_condition(True)
            .with_flush_immediately(True)]

    # f2 = get_error_cb(model, Xs, Ys, error_func, full=True)
    # tasks += [
    #     mon.ScalarFuncToTensorBoardTask(fw, f2, "error_full")
    #         .with_name('error_full')
    #         .with_condition(periodic_slow())
    #         .with_exit_condition(True)
    #         .with_flush_immediately(True)]

    print("# tasks:", len(tasks))
    return tasks


@ex.capture
def convgp_setup_optimizer(model, global_step, lr_cfg):
    if lr_cfg['decay'] == "custom":
        print("Custom decaying lr")
        lr = lr_cfg['lr'] * 1.0 / (1 + global_step // 5000 / 3)
    else:
        lr = lr_cfg['lr']
    return gpflow.train.AdamOptimizer(lr)


@ex.capture
def convgp_fit(train_data, test_data, iterations):
    session = gpflow.get_default_session()
    step = mon.create_global_step(session)
    model = convgp_setup_model(train_data)
    model.compile()
    optimizer = convgp_setup_optimizer(model, step)
    monitor_tasks = convgp_monitor_tasks(train_data, model, optimizer)
    monitor = mon.Monitor(monitor_tasks, session, step, print_summary=True)
    with monitor:
        optimizer.minimize(model,
                           step_callback=monitor,
                           maxiter=iterations,
                           global_step=step)
    convgp_finish(train_data, test_data, model)


def convgp_save(model):
    filename = experiment_path() + f'/convgp.gpflow'
    save_gpflow_model(filename, model)
    print(f"Model saved at {filename}")


@ex.capture
def convgp_finish(train_data, test_data, model, dataset):
    X, Y = train_data
    Xs, Ys = test_data
    error_func = calc_binary_error if dataset == "mnist01" else calc_multiclass_error
    error_func = get_error_cb(model, Xs, Ys, error_func, full=True)
    print(f"Error test: {error_func()}")
    print(f"Error train: {error_func()}")
    convgp_save(model)


######
## CNN
######


def cnn_monitor_callbacks():
    path = experiment_path()
    filename = path + '/cnn.{epoch:02d}-{val_acc:.2f}.h5'
    cbs = []
    cbs.append(ModelCheckpoint(filename, verbose=1, period=10))
    cbs.append(TensorBoard(path))
    return cbs


def cnn_setup_model():
    def MaxPool():
        return MaxPool2D(pool_size=(2, 2), strides=(2, 2))

    def Conv(num_kernels):
        return Conv2D(num_kernels, (5, 5), (1, 1),
                      activation='relu', padding='same',
                      kernel_initializer=truncated_normal(stddev=0.1))

    def FullyConnected(num_outputs, activation=None):
        return Dense(num_outputs,
                     activation=activation,
                     bias_initializer=constant(0.1),
                     kernel_initializer=truncated_normal(stddev=0.1))

    nn = Sequential()
    nn.add(Conv(32))
    nn.add(MaxPool())
    nn.add(Conv(64))
    nn.add(MaxPool())
    nn.add(Flatten())
    nn.add(FullyConnected(1024, activation='relu'))
    nn.add(Dropout(0.5))
    nn.add(FullyConnected(10, activation='softmax'))
    return nn


@ex.capture
def cnn_fit(train_data, test_data, batch_size, iterations):
    x, y = train_data
    model = cnn_setup_model()
    iters_per_epoch = x.shape[0] // batch_size
    epochs = iterations // iters_per_epoch
    callbacks = cnn_monitor_callbacks()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['mse', 'accuracy'])

    model.fit(x, y,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              validation_data=test_data)

    xt, yt = test_data
    test_metrics = model.evaluate(xt, yt, batch_size=batch_size)
    print(f"Test metrics: {test_metrics}")


@ex.capture
@ex.automain
def main(model_type):
    train_data, test_data = get_data()
    if model_type == "cnn":
        cnn_fit(train_data, test_data)
    else:
        convgp_fit(train_data, test_data)
    return 0
