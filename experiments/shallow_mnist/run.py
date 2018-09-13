import os

import gpflow
import gpflow.training.monitor as mon
import numpy as np
from observations import mnist, cifar10
from sacred import Experiment
from utils import get_error_cb, calc_multiclass_error, calc_binary_error

import gpflux

SUCCESS = 0
NAME = "mnist"
ex = Experiment(NAME)


@ex.config
def config():
    dataset = "mnist"
    # number of inducing points
    M = 1000
    # adam learning rate
    adam_lr = "decay"
    # training iterations
    iterations = int(50000)
    # patch size
    patch_shape = [5, 5]
    # path to save results
    basepath = "./"
    # minibatch size
    minibatch_size = 100

    base_kern = "RBF"

    # weighted sum
    with_weights = True

    # indexing
    with_indexing = True

    # init patches
    init_patches = "patches-unique"  # 'patches', 'random'

    restore = False

    # print hz
    hz = 50
    hz_slow = 500


@ex.capture
def data(basepath, dataset):
    def general_preprocess(X, Y, Xs, Ys):
        Y = Y.astype(int)
        Ys = Ys.astype(int)
        Y = Y.reshape(-1, 1)
        Ys = Ys.reshape(-1, 1)
        return X, Y, Xs, Ys

    def preprocess_mnist01(X, Y, Xs, Ys):
        def filter_01(X, Y):
            lbls01 = np.logical_or(Y == 0, Y == 1).flatten()
            return X[lbls01, :], Y[lbls01, :]

        X, Y = filter_01(X, Y)
        Xs, Ys = filter_01(Xs, Ys)
        return X, Y, Xs, Ys

    def preprocess_cifar(X, Y, Xs, Ys):
        def rgb2gray(rgb):
            rgb = np.transpose(rgb, [0, 2, 3, 1])
            proj_matrix = [0.299, 0.587, 0.114]
            return np.dot(rgb[..., :3], proj_matrix)

        X = rgb2gray(X)
        Xs = rgb2gray(Xs)
        X = X.reshape(-1, 32 ** 2)
        Xs = Xs.reshape(-1, 32 ** 2)
        return X, Y, Xs, Ys

    def preprocess_full(X, Y, Xs, Ys):
        return X, Y, Xs, Ys

    data_dict = {"mnist01": mnist,
                 "mnist": mnist,
                 "cifar": cifar10}
    preprocess_dict = {"mnist01": preprocess_mnist01,
                       "mnist": preprocess_full,
                       "cifar": preprocess_cifar}

    data_func = data_dict[dataset]
    path = os.path.join(basepath, "data")
    (X, Y), (Xs, Ys) = data_func(path)
    X, Y, Xs, Ys = general_preprocess(X, Y, Xs, Ys)
    preprocess_func = preprocess_dict[dataset]
    X, Y, Xs, Ys = preprocess_func(X, Y, Xs, Ys)

    alpha = 255.0
    return X / alpha, Y, Xs / alpha, Ys


@ex.capture
def experiment_name(adam_lr, M, minibatch_size, dataset,
                    base_kern, init_patches, patch_shape,
                    with_weights, with_indexing):
    args = np.array(
        [
            "conv_ops",
            dataset,
            "W", with_weights,
            "Ti", with_indexing,
            "init_patches", init_patches,
            "kern", base_kern,
            "adam", adam_lr,
            "M", M,
            "minibatch_size", minibatch_size,
            "patch", patch_shape[0],
        ])
    return "_".join(args.astype(str))


@ex.capture
def restore_session(session, restore, basepath):
    model_path = os.path.join(basepath, NAME, experiment_name())
    if restore and os.path.isdir(model_path):
        mon.restore_session(session, model_path)
        print("Model restored")


@gpflow.defer_build()
@ex.capture
def setup_model(X, Y, minibatch_size, patch_shape, M, dataset, base_kern,
                init_patches, basepath, restore, with_weights, with_indexing):
    if dataset == "mnist01":
        like = gpflow.likelihoods.Bernoulli()
        num_filters = 1
    else:
        like = gpflow.likelihoods.SoftMax(10)
        num_filters = 10

    H = int(X.shape[1] ** .5)

    if init_patches == "random":
        patches = gpflux.init.NormalInitializer()
    else:
        unique = init_patches == "patches-unique"
        patches = gpflux.init.PatchSamplerInitializer(
            X[:100], width=H, height=H, unique=unique)

    layer0 = gpflux.layers.WeightedSumConvLayer(
        [H, H], M, patch_shape,
        num_latents=num_filters,
        with_indexing=with_indexing,
        with_weights=with_weights,
        patches_initializer=patches)

    # init kernel
    if with_indexing:
        layer0.kern.index_kernel.variance = 25.0
        layer0.kern.index_kernel.lengthscales = 3.0
    layer0.kern.basekern.variance = 25.0
    layer0.kern.basekern.lengthscales = 1.2

    # break symmetry in variational parameters
    layer0.q_sqrt = layer0.q_sqrt.read_value()
    layer0.q_mu = np.random.randn(*(layer0.q_mu.read_value().shape))

    model = gpflux.DeepGP(X, Y,
                          layers=[layer0],
                          likelihood=like,
                          batch_size=minibatch_size,
                          name="my_deep_gp")
    return model


@ex.capture
def setup_monitor_tasks(Xs, Ys, model, optimizer,
                        hz, hz_slow, basepath, dataset, adam_lr):
    tb_path = os.path.join(basepath, NAME, "tensorboards", experiment_name())
    model_path = os.path.join(basepath, NAME, experiment_name())
    fw = mon.LogdirWriter(tb_path)

    # print_error = mon.CallbackTask(error_cb)\
    # .with_name('error')\
    # .with_condition(mon.PeriodicIterationCondition(hz))\
    # .with_exit_condition(True)
    tasks = []

    if adam_lr == "decay":
        def lr(*args, **kwargs):
            sess = model.enquire_session()
            return sess.run(optimizer._optimizer._lr)

        tasks += [
            mon.ScalarFuncToTensorBoardTask(fw, lr, "lr")
                .with_name('lr')
                .with_condition(mon.PeriodicIterationCondition(hz))
                .with_exit_condition(True)
                .with_flush_immediately(True)]

    tasks += [
        mon.CheckpointTask(model_path)
            .with_name('saver')
            .with_condition(mon.PeriodicIterationCondition(hz_slow))]

    tasks += [
        mon.ModelToTensorBoardTask(fw, model)
            .with_name('model_tboard')
            .with_condition(mon.PeriodicIterationCondition(hz))
            .with_exit_condition(True)
            .with_flush_immediately(True)]

    tasks += [
        mon.PrintTimingsTask().with_name('print')
            .with_condition(mon.PeriodicIterationCondition(hz))
            .with_exit_condition(True)]

    error_func = calc_binary_error if dataset == "mnist01" \
        else calc_multiclass_error

    f1 = get_error_cb(model, Xs, Ys, error_func)
    tasks += [
        mon.ScalarFuncToTensorBoardTask(fw, f1, "error")
            .with_name('error')
            .with_condition(mon.PeriodicIterationCondition(hz))
            .with_exit_condition(True)
            .with_flush_immediately(True)]

    f2 = get_error_cb(model, Xs, Ys, error_func, full=True)
    tasks += [
        mon.ScalarFuncToTensorBoardTask(fw, f2, "error_full")
            .with_name('error_full')
            .with_condition(mon.PeriodicIterationCondition(hz_slow))
            .with_exit_condition(True)
            .with_flush_immediately(True)]

    print("# tasks:", len(tasks))
    return tasks


@ex.capture
def setup_optimizer(model, global_step, adam_lr):
    if adam_lr == "decay":
        print("decaying lr")
        lr = 0.01 * 1.0 / (1 + global_step // 5000 / 3)
        # lr = tf.train.exponential_decay(learning_rate=0.01,
        #                                 global_step=global_step,
        #                                 decay_steps=500,
        #                                 decay_rate=.95,
        #                                 staircase=False)
    else:
        lr = adam_lr

    return gpflow.train.AdamOptimizer(lr)


@ex.capture
def run(model, session, global_step, monitor_tasks, optimizer, iterations):
    monitor = mon.Monitor(monitor_tasks, session, global_step, print_summary=True)

    with monitor:
        optimizer.minimize(model,
                           step_callback=monitor,
                           maxiter=iterations,
                           global_step=global_step)
    return model.compute_log_likelihood()


def _save(model, filename):
    gpflow.Saver().save(filename, model)
    print("model saved")


@ex.capture
def finish(X, Y, Xs, Ys, model, dataset, basepath):
    print(model)
    error_func = calc_binary_error if dataset == "mnist01" \
        else calc_multiclass_error
    error_func = get_error_cb(model, Xs, Ys, error_func, full=True)
    print("error test", error_func())
    print("error train", error_func())

    fn = os.path.join(basepath, NAME) + "/" + experiment_name() + ".gpflow"
    _save(model, fn)


@ex.capture
def trace_run(model, sess, M, minibatch_size, adam_lr):
    name = "M_{}_N_{}_pyfunc_gpu".format(M, minibatch_size)
    # name =  "test"
    from utils import trace

    with sess:
        like = model.likelihood_tensor
        trace(like, sess, "trace_likelihood_{}.json".format(name))

        adam_opt = gpflow.train.AdamOptimizer(learning_rate=0.01)
        adam_step = adam_opt.make_optimize_tensor(model, session=sess)
        trace(adam_step, sess, "trace_adam_{}.json".format(name))


@ex.automain
def main():
    X, Y, Xs, Ys = data()

    model = setup_model(X, Y)
    model.compile()
    sess = model.enquire_session()
    step = mon.create_global_step(sess)

    restore_session(sess)

    print(model)
    print("X", np.min(X), np.max(X))
    print("before optimisation ll", model.compute_log_likelihood())

    optimizer = setup_optimizer(model, step)
    monitor_tasks = setup_monitor_tasks(Xs, Ys, model, optimizer)
    ll = run(model, sess, step, monitor_tasks, optimizer)
    print("after optimisation ll", ll)

    finish(X, Y, Xs, Ys, model)

    return SUCCESS
