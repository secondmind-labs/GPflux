import os
import gpflow
import gpflow.training.monitor as mon
import gpflux
import numpy as np
import tensorflow as tf

from sacred import Experiment
from observations import mnist, cifar10

from utils import get_error_cb, calc_multiclass_error, calc_binary_error

SUCCESS = 0
NAME = "mnist_new3"
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

    # init patches
    init_patches = "patches-unique" # 'patches', 'random'

    restore = False

    # print hz
    hz = 10
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
            return np.dot(rgb[...,:3], proj_matrix)

        X = rgb2gray(X)
        Xs = rgb2gray(Xs)
        X = X.reshape(-1, 32**2)
        Xs = Xs.reshape(-1, 32**2)
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
    return X/alpha, Y, Xs/alpha, Ys



@ex.capture
def experiment_name(adam_lr, M, minibatch_size, dataset,
                    base_kern, init_patches, patch_shape):
    args = np.array(
        [
            "deep",
            dataset,
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
                init_patches, basepath, restore):

    H = int(X.shape[1]**.5)

    ## layer 0: indexed conv patch 5x5
    patches0 = gpflux.init.PatchSamplerInitializer(X[:100], width=H, height=H, unique=True)
    patches0 = patches0.sample([M, 5, 5])
    indices0 = np.random.randint(0, 24, size=(M, 2)).astype(np.float64)
    feature0 = gpflux.convolution.IndexedInducingPatch(patches0, indices0)

    conv_kernel0 = gpflow.kernels.RBF(25, variance=30.0, lengthscales=1.2)
    kernel0 = gpflux.convolution.ConvKernel(conv_kernel0,
                                            img_size=[28, 28],
                                            patch_shape=[5, 5],
                                            pooling=2,
                                            with_indexing=True)
    kernel0.index_kernel.variance=30.0
    layer0 = gpflux.layers.GPLayer(kernel0, feature0, num_latents=1)


    ## layer 1: indexed conv patch 3x3
    patches1 = np.random.randn(M, np.prod([3, 3]))
    indices1 = np.random.randint(0, 10, size=(M, 2)).astype(np.float64)
    feature1 = gpflux.convolution.IndexedInducingPatch(patches1, indices1)

    conv_kernel1 = gpflow.kernels.RBF(9, variance=10.0, lengthscales=1.2)
    kernel1 = gpflux.convolution.ConvKernel(conv_kernel1,
                                            img_size=[12, 12],
                                            patch_shape=[3, 3],
                                            pooling=2,
                                            with_indexing=True)
    kernel1.index_kernel.variance=10.0
    layer1 = gpflux.layers.GPLayer(kernel1, feature1, num_latents=1)

    layer1.q_sqrt = layer1.q_sqrt.read_value() * 0.1
    layer1.q_mu = np.random.randn(*(layer1.q_mu.read_value().shape))

    ### Equivalent layers in written with ConvLayers
    # layer0 = gpflux.layers.ConvLayer([28, 28], [12, 12], M, [5, 5],
    #                                  num_latents=1,
    #                                  pooling=2,
    #                                  with_indexing=True)
    # layer0.q_sqrt = layer0.q_sqrt.read_value() * 0.1
    # layer0.q_mu = np.random.randn(*(layer0.q_mu.read_value().shape))
    # layer0.kern.index_kernel.variance = 30.0
    # layer0.kern.basekern.variance = 30.0
    # layer0.kern.basekern.lengthscales = 1.5


    # layer1 = gpflux.layers.ConvLayer([12, 12], [5, 5], M, [3, 3],
    #                                  num_latents=1,
    #                                  pooling=2,
    #                                  with_indexing=True)
    # layer1.q_sqrt = layer1.q_sqrt.read_value() * 0.1
    # layer1.q_mu = np.random.randn(*(layer1.q_mu.read_value().shape))
    # layer1.kern.index_kernel.variance = 30.0
    # layer1.kern.basekern.variance = 30.0
    # layer1.kern.basekern.lengthscales = 1.5


    ## layer 2: fully connected
    feature2 = gpflow.features.InducingPoints(np.random.randn(M, 25))
    kernel2 = gpflow.kernels.RBF(25)
    layer2 = gpflux.layers.GPLayer(kernel2, feature2, num_latents=10)

    layer2.q_sqrt = layer2.q_sqrt.read_value() * 0.1
    layer2.q_mu = np.random.randn(*(layer2.q_mu.read_value().shape))


    if dataset == "mnist01":
        like = gpflow.likelihoods.Bernoulli()
    else:
        like = gpflow.likelihoods.SoftMax(10)

    model = gpflux.DeepGP(X, Y,
                          layers=[layer0, layer1, layer2],
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

        tasks += [\
              mon.ScalarFuncToTensorBoardTask(fw, lr, "lr")\
              .with_name('lr')\
              .with_condition(mon.PeriodicIterationCondition(hz))\
              .with_exit_condition(True)\
              .with_flush_immediately(True)]

    tasks += [\
        mon.CheckpointTask(model_path)\
        .with_name('saver')\
        .with_condition(mon.PeriodicIterationCondition(hz_slow))]

    tasks += [\
        mon.ModelToTensorBoardTask(fw, model)\
        .with_name('model_tboard')\
        .with_condition(mon.PeriodicIterationCondition(hz))\
        .with_exit_condition(True)\
        .with_flush_immediately(True)]

    tasks += [\
        mon.PrintTimingsTask().with_name('print')\
        .with_condition(mon.PeriodicIterationCondition(hz))\
        .with_exit_condition(True)]

    error_func = calc_binary_error if dataset == "mnist01" \
                    else calc_multiclass_error

    f1 = get_error_cb(model, Xs, Ys, error_func)
    tasks += [\
          mon.ScalarFuncToTensorBoardTask(fw, f1, "error")\
          .with_name('error')\
          .with_condition(mon.PeriodicIterationCondition(hz))\
          .with_exit_condition(True)\
          .with_flush_immediately(True)]

    f2 = get_error_cb(model, Xs, Ys, error_func, full=True)
    tasks += [\
          mon.ScalarFuncToTensorBoardTask(fw, f2, "error_full")\
          .with_name('error_full')\
          .with_condition(mon.PeriodicIterationCondition(hz_slow))\
          .with_exit_condition(True)\
          .with_flush_immediately(True)]

    print("# tasks:", len(tasks))
    return tasks


@ex.capture
def setup_optimizer(model, global_step, adam_lr):

    if adam_lr == "decay":
        print("decaying lr")
        lr = tf.train.exponential_decay(learning_rate=0.01,
                                        global_step=global_step,
                                        decay_steps=-10000,
                                        decay_rate=10,
                                        staircase=True)
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

    name =  "M_{}_N_{}_pyfunc_gpu".format(M, minibatch_size)
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

    # restore_session(sess)

    print(model)
    # print("X", np.min(X), np.max(X))
    print("before optimisation ll", model.compute_log_likelihood())

    # return 0

    optimizer = setup_optimizer(model, step)
    monitor_tasks = setup_monitor_tasks(Xs, Ys, model, optimizer)
    ll = run(model, sess, step,  monitor_tasks, optimizer)
    print("after optimisation ll", ll)

    finish(X, Y, Xs, Ys, model)

    return SUCCESS
