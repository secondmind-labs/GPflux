# pylint: disable=E1120,W0612

import datetime
from pathlib import Path
from typing import Any, List, Tuple

import datasets
import gp_utils
import numpy as np
import tensorflow as tf
import train_utils
from sacred import Experiment

import gpflow
from gpflow import settings
from gpflow.training import monitor as mon

NAME = "classification"
ex = Experiment(NAME)


@ex.config
def config():
    name = "gp"
    seed = None
    basepath = '.'
    dataset = 'mnist'
    date = datetime.datetime.now().strftime('%d-%H-%M')
    float_type = 'float64'

    num_inducing_points = 100
    batch_size = 32
    total_epochs = 600

    feature = ('ip', {})
    kernel = ('sirot', {})
    likelihood = ('softmax', {})

    hz = {
        'frequent': 50,
        'epoch': 390
    }

    unfreeze = {
        # 'kernel': 12000 # step
    }

    freeze = {
        # 'kernel': 12000 # step
    }

    lr = dict(
        lr = 0.003,
        mode = 'step_decay',  # custom_step_decay, step_decay, constant
        rate = 0.95,
        num_steps = hz['epoch'],
    )


def experiment_path(basepath: str, name: str, kwargs: List[Tuple[Any, Any]]) -> str:
    assert isinstance(name, str)
    for (k, v) in kwargs:
        name += "-{0}_{1}".format(k, v)
    return str(Path(basepath, name))


@ex.capture
def set_default_float_type(float_type):
    types = {'float32': np.float32, 'float64': np.float64}
    gpflow.settings.dtypes['float_type'] = types[float_type]


@ex.capture
def get_kernel(kernel, dataset):
    _, kern_args = kernel
    if 'input_dim' not in kern_args:
        kern_args['input_dim'] = np.prod(datasets.input_shape(dataset))
        kernel[1] = kern_args
    return gp_utils.get_kernel(kernel)


@ex.capture
def get_likelihood(likelihood):
    return gp_utils.get_likelihood(likelihood)


@ex.capture
def get_feature(values, feature, num_inducing_points, seed):
    return gp_utils.get_feature(feature, values, num_inducing_points, seed=seed)


@ex.capture
def num_classes(dataset):
    return {
        'rotmnist': 10,
        'cifar10': 10,
        'mnist': 10,
        'cifar100': 100
    }[dataset]


@ex.capture
def num_latent(dataset, likelihood):
    return num_classes()


@ex.capture
def experiment_log_path(name, basepath,
        date, lr, likelihood, kernel,
        freeze, unfreeze, feature, num_inducing_points):
    """Experiment path name. Folder where TensorBoard, saved models and logs are stored."""

    path = str(Path(basepath, f"{name}-{date}"))

    name_args = []
    name_args.append(("num_inducing_points", num_inducing_points))

    for gp_comp_key, gp_comp in [('lik', likelihood), ('kern', kernel), ('feat', feature)]:
        comp_name, comp_args = gp_comp
        name_args.append((gp_comp_key, comp_name))
        for kv in comp_args.items():
            name_args.append(kv)

    lr = lr.copy()
    lr_mode = lr.pop('mode')
    name_args.append(('lr_mode', lr_mode))
    for kv in lr.items():
        name_args.append(kv)

    def freezing_to_string(d):
        values = ""
        for part, step in d.items():
            step = 0 if step is None else step
            if values != "":
                values += "_"
            values += "{part}{step}".format(part=part, step=step)
        return values

    if freeze is not None and len(freeze) > 0:
        name_args.append(('freeze', freezing_to_string(freeze)))

    if unfreeze is not None and len(unfreeze) > 0:
        name_args.append(('unfreeze', freezing_to_string(unfreeze)))

    return experiment_path(path, name, name_args)


@ex.capture
def get_data(dataset):
    (x, y), (xt, yt) = datasets.get_dataset(dataset)
    return (x, y), (xt, yt)


@ex.capture
def preprocess_data(data, likelihood):
    (x, y), (xt, yt) = data
    N = x.shape[0]
    Nt = xt.shape[0]
    if likelihood[0] == "gaussian":
        # If the likelihood is Gaussian, we need one-hot encoded classes?
        y = np.eye(num_classes(), dtype=settings.float_type)[y.reshape(-1)]
        yt = np.eye(num_classes(), dtype=settings.float_type)[yt.reshape(-1)]
    return (x.reshape([N, -1]), y), (xt.reshape([Nt, -1]), yt)


@ex.capture
def make_optimizer(global_step, lr):
    if lr['mode'] == "step_decay":
        power = tf.floor((1 + global_step) / lr['num_steps'])
        lr_scheme = lr['lr'] * tf.pow(np.array(lr['rate'], dtype=np.float64), power)
    elif lr['mode'] == "custom_step_decay":
        denom = 1 + lr['rate'] * tf.cast(global_step // lr['num_steps'], settings.float_type)
        lr_scheme = lr['lr'] / denom
    else:
        lr_scheme = lr['lr']

    return gpflow.train.AdamOptimizer(lr_scheme)


@ex.capture
def monitor_tasks(model, data, optimizer, name, likelihood, batch_size, hz):
    train_data, valid_data = data
    writer = mon.LogdirWriter(experiment_log_path())

    periodic = mon.PeriodicIterationCondition
    tasks = []

    lr_cb = train_utils.make_learning_rate_cb(optimizer)

    onehot = likelihood[0] == "gaussian"
    classes = num_classes()

    valid_metrics_cb = train_utils.make_metrics_cb(model, valid_data, 'valid', batch_size, classes, is_onehot=onehot)
    train_metrics_cb = train_utils.make_metrics_cb(model, train_data, 'train', batch_size, classes, is_onehot=onehot)

    frequent_hz = hz['frequent']
    epoch_hz = hz['epoch']

    tasks.append(
        mon.ScalarFuncToTensorBoardTask(writer, lr_cb, 'learning-rate')
        .with_name('learning-rate')
        .with_exit_condition(1)
        .with_flush_immediately(1)
        .with_condition(periodic(frequent_hz)))

    tasks.append(
        mon.PrintTimingsTask()
        .with_condition(periodic(frequent_hz))
        .with_exit_condition(1))

    tasks.append(
        mon.ModelToTensorBoardTask(writer, model)
        .with_name(name)
        .with_condition(periodic(frequent_hz))
        .with_exit_condition(1)
        .with_flush_immediately(1))

    tasks.append(
        train_utils.ScalarsToTensorBoardTask(writer, valid_metrics_cb)
        .with_exit_condition(1)
        .with_condition(periodic(epoch_hz)))

    tasks.append(
        train_utils.ScalarsToTensorBoardTask(writer, train_metrics_cb)
        .with_exit_condition(1)
        .with_condition(periodic(epoch_hz)))

    return tasks


def freezing(model, freeze_or_unfreeze, step, options) -> bool:
    train = False if freeze_or_unfreeze == "freeze" else True
    success = False
    for part, on_step in options.items():
        if on_step != step:
            continue
        success = True
        if part == 'kernel':
            model.kern.trainable = train
    return success


@ex.capture
def do_freeze(model, step, freeze):
    return freezing(model, "freeze", step, freeze)


@ex.capture
def do_unfreeze(model, step, unfreeze):
    return freezing(model, "unfreeze", step, unfreeze)


@ex.capture
def build_model(data, batch_size):
    (x, y), (xt, yt) = data
    X_num = x.shape[0]
    features = get_feature(x)
    kernel = get_kernel()
    likelihood = get_likelihood()
    svgp = gpflow.models.SVGP(x, y, kernel, likelihood, features,
                              num_data=X_num, num_latent=num_latent(),
                              minibatch_size=batch_size, name='SVGP')
    do_freeze(svgp, 0)
    do_unfreeze(svgp, 0)
    return svgp


@ex.capture
def iterations_per_epoch(num_data, batch_size):
    return num_data // batch_size


def current_step(num_data, epoch, iteration):
    return epoch * iterations_per_epoch(num_data) + iteration


@ex.capture
def fit_gp_model(model, data, total_epochs, batch_size):
    sess = gpflow.get_default_session()
    global_step = mon.create_global_step(sess)
    inc_global_step = tf.assign_add(global_step, 1)
    opt = make_optimizer(global_step)
    opt_op = train_utils.make_optimize_operation(model, opt, sess)

    num_data = data[0][0].shape[0]
    iters_per_epoch = iterations_per_epoch(num_data)
    maxiter = iters_per_epoch * total_epochs

    tasks = monitor_tasks(model, data, opt)
    with mon.Monitor(tasks, sess, global_step, print_summary=True) as m:
        for epoch in range(total_epochs):
            for iteration in range(iters_per_epoch):
                step = current_step(num_data, epoch, iteration) + 1
                unfreeze_success = do_unfreeze(model, step)
                freeze_success = do_freeze(model, step)
                if unfreeze_success or freeze_success:
                    opt_op = train_utils.make_optimize_operation(model, opt, sess)
                _, step = sess.run([opt_op, inc_global_step])
                m(step)


@ex.automain
def main():
    set_default_float_type()
    data = preprocess_data(get_data())
    model = build_model(data)
    fit_gp_model(model, data)
    return 0