import os
import gpflow
import gpflow.training.monitor as mon
import gpflux
import numpy as np
import tensorflow as tf

from sacred import Experiment
from observations import mnist

from utils import get_error_cb, calc_multiclass_error, calc_binary_error

SUCCESS = 0
NAME = "mnist-indexed-conv2"
ex = Experiment(NAME)


@ex.config
def config():
    dataset = "full"
    # number of inducing points
    M = 500
    # adam learning rate
    adam_lr = 0.1
    # training iterations
    iterations = int(2e3)
    # patch size
    patch_size = [5, 5]
    # path to save results
    basepath = "/mnt/vincent/"
    # minibatch size
    minibatch_size = 100

    base_kern = "RBF"

    # Convolutional kernel or not
    conv = True

    # first fixed
    fixed_first = False

    # init patches
    init_patches = "patches-unique" # 'patches', 'random'

    # print hz
    hz = 10
    hz_slow = 500


@ex.capture
def data(basepath, dataset, normalize=True):
    path = os.path.join(basepath, "data")
    (X, Y), (Xs, Ys) = mnist(path)
    Y = Y.astype(int)
    Ys = Ys.astype(int)
    Y = Y.reshape(-1, 1)
    Ys = Ys.reshape(-1, 1)
    alpha = 255. if normalize else 1.

    if dataset == "01":
        def filter_01(X, Y):
            lbls01 = np.logical_or(Y == 0, Y == 1).flatten()
            return X[lbls01, :], Y[lbls01, :]

        X, Y = filter_01(X, Y)
        Xs, Ys = filter_01(Xs, Ys)
        return X/alpha, Y, Xs/alpha, Ys

    elif dataset == "full":
        return X/alpha, Y, Xs/alpha, Ys

    else:
        raise ValueError("dataset {} is unknown".format(dataset))


@ex.capture
def experiment_name(adam_lr, M, minibatch_size, dataset,
                    base_kern, conv, fixed_first, init_patches,
                    patch_size):
    args = np.array(
        [
            "norm",
            dataset,
            "conv", conv,
            "fixed_first", fixed_first,
            "init_patches", init_patches,
            "kern", base_kern,
            "adam", adam_lr,
            "M", M,
            "minibatch_size", minibatch_size,
            "patch", patch_size[0],
        ])
    return "_".join(args.astype(str))


@ex.capture
def setup_model(X, Y, minibatch_size, patch_size, M, dataset, base_kern,
                conv, init_patches):

    if dataset == "01":
        like = gpflow.likelihoods.Bernoulli()
        num_filters = 1
    else:
        like = gpflow.likelihoods.SoftMax(10)
        num_filters = 10

    if base_kern == "RBF":
        kern = gpflow.kernels.RBF
    else:
        kern = gpflow.kernels.ArcCosine

    H = int(X.shape[1]**.5)

    if conv:
        print("Conv")
        if init_patches == "random":
            patches = gpflux.init.NormalInitializer()
        else:
            unique = init_patches == "patches-unique"
            patches = gpflux.init.PatchSamplerInitializer(X[:100], width=H, height=H,
                                                          unique=unique)

        layer0 = gpflux.layers.PoolingIndexedConvLayer(
                    [H, H], M, patch_size,
                    num_filters=num_filters,
                    patches_initializer=patches,
                    base_kernel_class=kern)

        # init kernel
        # layer0.kern.index_kernel.variance = 5.0
        # layer0.kern.conv_kernel.basekern.variance = 0.5
        layer0.kern.index_kernel.lengthscales = 3.0
        layer0.kern.conv_kernel.basekern.lengthscales = 0.5
        layer0.kern.set_trainable(False)

        # break symmetry in variational parameters
        layer0.q_sqrt = layer0.q_sqrt.read_value() * .5
        layer0.q_mu = np.random.randn(*(layer0.q_mu.read_value().shape)) * .5

    else:
        print("No conv")
        feat = gpflow.features.InducingPoints(X[:M].copy())
        kern = gpflow.kernels.Polynomial(X.shape[1], degree=9.0) + \
                 gpflow.kernels.RBF(X.shape[1]) + \
                 gpflow.kernels.White(X.shape[1], 1e-1)
        # kern = kern(X.shape[1])
        layer0 = gpflux.layers.GPLayer(kern, feat, num_latents=num_filters)
        layer0.q_sqrt = layer0.q_sqrt.read_value() * .5
        layer0.q_mu = np.random.randn(*(layer0.q_mu.read_value().shape))


    model = gpflux.DeepGP(X, Y,
                          layers=[layer0],
                          likelihood=like,
                          batch_size=minibatch_size)
    return model

@ex.capture
def setup_monitor_tasks(Xs, Ys, model, hz, hz_slow, basepath, dataset):
    tb_path = os.path.join(basepath, NAME, "tensorboard", experiment_name())
    fw = mon.LogdirWriter(tb_path)

    # print_error = mon.CallbackTask(error_cb)\
        # .with_name('error')\
        # .with_condition(mon.PeriodicIterationCondition(hz))\
        # .with_exit_condition(True)
    tb_model = mon.ModelToTensorBoardTask(fw, model)\
        .with_name('model_tboard')\
        .with_condition(mon.PeriodicIterationCondition(hz))\
        .with_exit_condition(True)\
        .with_flush_immediately(True)

    print_task = mon.PrintTimingsTask().with_name('print')\
        .with_condition(mon.PeriodicIterationCondition(hz))\
        .with_exit_condition(True)

    error_func = calc_binary_error if dataset == "01" else calc_multiclass_error

    f1 = get_error_cb(model, Xs, Ys, error_func)
    tb_error = mon.ScalarFuncToTensorBoardTask(fw, f1, "error")\
          .with_name('error')\
          .with_condition(mon.PeriodicIterationCondition(hz))\
          .with_exit_condition(True)\
          .with_flush_immediately(True)

    f2 = get_error_cb(model, Xs, Ys, error_func, full=True)
    tb_error_full = mon.ScalarFuncToTensorBoardTask(fw, f2, "error_full")\
          .with_name('error_full')\
          .with_condition(mon.PeriodicIterationCondition(hz_slow))\
          .with_exit_condition(True)\
          .with_flush_immediately(True)

    return [tb_error, tb_error_full, tb_model, print_task]


@ex.capture
def setup_optimisation_and_run(model, tasks, adam_lr, iterations, fixed_first):
    session = model.enquire_session()
    global_step = mon.create_global_step(session)

    monitor = mon.Monitor(tasks, session, global_step, print_summary=True)
    optimiser = gpflow.train.AdamOptimizer(adam_lr)

    print(experiment_name())

    with monitor:
        if fixed_first:
            optimiser.minimize(model,
                               step_callback=monitor,
                               maxiter=100,
                               global_step=global_step)

        model.set_trainable(True)
        optimiser.minimize(model,
                           step_callback=monitor,
                           maxiter=iterations,
                           global_step=global_step)

@ex.capture
def finish(X, Y, Xs, Ys, model, dataset):
    print(model)
    error_func = calc_binary_error if dataset == "01" else calc_multiclass_error
    error_func = get_error_cb(model, Xs, Ys, error_func, full=True)
    print("error test", error_func())
    print("error train", error_func())



# def trace(model):
#     from tensorflow.python.client import timeline

#     sess = model.enquire_session()
#     # adam_opt = gpflow.train.AdamOptimizer(learning_rate=0.01)
#     # adam_step = adam_opt.make_optimize_tensor(model, session=sess)
#     like = model.likelihood_tensor

#     with sess:
#         # add additional options to trace the session execution
#         options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#         run_metadata = tf.RunMetadata()
#         sess.run(like, options=options, run_metadata=run_metadata)

#         # Create the Timeline object, and write it to a json file
#         fetched_timeline = timeline.Timeline(run_metadata.step_stats)
#         chrome_trace = fetched_timeline.generate_chrome_trace_format()
#         with open('timeline_likelihood.json', 'w') as f:
#             f.write(chrome_trace)

@ex.automain
def main():
    X, Y, Xs, Ys = data()

    model = setup_model(X, Y)

    Z = model.layers[0].feature.patches.Z.read_value()
    print("X")
    print(np.min(X))
    print(np.max(X))

    print("before optimisation ll", model.compute_log_likelihood())
    print(model)
    monitor_tasks = setup_monitor_tasks(Xs, Ys, model)
    setup_optimisation_and_run(model, monitor_tasks)
    print("after optimisation ll", model.compute_log_likelihood())
    finish(X, Y, Xs, Ys, model)
    return SUCCESS

    # sess = model.enquire_session()
    # train(model, sess)
    # save(model)

    # return model.compute_log_likelihood()


# base_kernel = gpflow.kernels.RBF(np.prod(patch_size)) # + gpflow.kernels.White(np.prod(patch_size), variance=0.01)
# conv_kernel = gpflux.convolution_kernel.ConvKernel(base_kernel, img_size_in, patch_size)
# index_kernel = gpflow.kernels.RBF(2, lengthscales=5., ARD=True) # + gpflow.kernels.White(2, variance=0.01)
# kern = gpflux.convolution_kernel.IndexedConvKernel(conv_kernel, index_kernel)

# inducing_patches_initializer = gpflux.init.PatchSamplerInitializer(X, width=W, height=H)
# inducing_patches = inducing_patches_initializer.sample([M, *patch_size])  # M x w x h
# inducing_patches = gpflux.inducing_patch.InducingPatch(inducing_patches.reshape([M, np.prod(patch_size)]))


# Z_indices = np.random.randint(0, H, size=(M, 2))
# inducing_indices = gpflow.features.InducingPoints(Z_indices)
# feat = gpflux.inducing_patch.IndexedInducingPatch(inducing_patches, inducing_indices)

# calc_error = calc_binary_error if Nc == 2 else calc_multiclass_error

# if Nc > 2:
#     like = gpflow.likelihoods.SoftMax(Nc)
#     num_latent = Nc
# else:
#     like = gpflow.likelihoods.Bernoulli()
#     num_latent = 1
# print(like.__class__.__name__)
# m = gpflow.models.SVGP(X, Y, kern,
#                        likelihood=like,
#                        feat=feat,
#                        num_latent=num_latent,
#                        minibatch_size=200)
# m.q_sqrt = m.q_sqrt.read_value() * 1e-3

# MAXITER = 1000
# print_task = mon.PrintTimingsTask().with_name('print')\
#     .with_condition(mon.PeriodicIterationCondition(10))\
#     .with_exit_condition(True)

# def cb(*args, **kwargs):
#     # m.anchor(m.enquire_session())
#     # print(m)
#     print("elbo", m.compute_log_likelihood())

# print_lml = mon.CallbackTask(cb)\
#     .with_name('lml')\
#     .with_condition(mon.PeriodicIterationCondition(50))\
#     .with_exit_condition(True)

# def cb2(*args, **kwargs):
#     Ns = 1000
#     print("error", calc_error(m, Xs[:Ns], Ys[:Ns], batchsize=50))

# print_error = mon.CallbackTask(cb2)\
#     .with_name('error')\
#     .with_condition(mon.PeriodicIterationCondition(50))\
#     .with_exit_condition(True)

# session = m.enquire_session()
# global_step = mon.create_global_step(session)

# print("before", m.compute_log_likelihood())
# print("before error", calc_error(m, Xs, Ys))

# optimiser = gpflow.train.AdamOptimizer(0.01)

# monitor = mon.Monitor([print_task, print_lml, print_error], session, global_step, print_summary=True)
# with monitor:
#     optimiser.minimize(m, step_callback=monitor, maxiter=MAXITER, global_step=global_step)

# print("after", m.compute_log_likelihood())
# print("after error", calc_error(m, Xs, Ys))

# print(m)
