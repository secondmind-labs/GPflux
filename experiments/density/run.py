import os

import numpy as np
import tensorflow as tf
from sacred import Experiment

import gpflow
import gpflow.training.monitor as mon
import gpflux
from data import fixed_binarized_mnist
from utils import calculate_nll

SUCCESS = 0
NAME = "FixBinMnist"
ex = Experiment(NAME)

class MoBernoulli(gpflow.likelihoods.Bernoulli):

    def predict_mean_from_f_full_output_cov(self, Fmu, Fvar):
        """
        Fmu: N x D
        Fvar: N x D x D
        """
        return gpflow.quadrature.full_gaussian_mc(self.conditional_mean, 100, Fmu, Fvar)

    def eval(self, P, Y):
        return gpflow.logdensities.bernoulli(Y, P)

@ex.config
def config():

    # number of inducing points
    num_inducing = 1000
    # adam learning rate
    adam_lr = "decay"
    # training iterations
    iterations = int(50000)
    # patch size
    patch_size = [5, 5]
    # path to save results
    basepath = "./"
    # minibatch size
    minibatch_size = 100

    restore = False

    # print hz
    hz = 10
    hz_slow = 500


@ex.capture
def data(basepath):

    path = os.path.join(basepath, "data")
    X, Xs = fixed_binarized_mnist(path)

    # X, m, s = normalize(X)
    # Xs, _, _ = normalize(Xs, m, s)

    return X, Xs


@ex.capture
def experiment_name(adam_lr, num_inducing, minibatch_size, patch_size):
    args = np.array(
        [
            "adam", adam_lr,
            "M", num_inducing,
            "minibatch_size", minibatch_size,
            "patch", patch_size[0],
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
def setup_model(X, minibatch_size, patch_size, num_inducing):

    assert patch_size[0] == patch_size[1]

    latent_dim = 5
    # H = int(X.shape[1]**.5)

    ## layer 0
    enc = gpflux.encoders.RecognitionNetwork(latent_dim, X.shape[1], [200, 200, 100])
    layer0 = gpflux.layers.LatentVariableConcatLayer(latent_dim, encoder=enc)

    ## layer 1
    kern_list = [gpflow.kernels.RBF(latent_dim) for _ in range(20)]
    W = np.random.randn(20, 1024)
    kern = gpflow.multioutput.kernels.SeparateMixedMok(kern_list, W.T)
    feat = gpflow.multioutput.features.MixedKernelSharedMof(
            gpflow.features.InducingPoints(np.random.randn(100, latent_dim)))
    layer1 = gpflux.layers.GPLayer(kern, feat, num_latents=20)

    ## layer 2: indexed conv patch 5x5
    layer2 = gpflux.layers.ConvLayer([32, 32], [28, 28], num_inducing, patch_size=[5, 5], with_indexing=True)

    layer2.kern.index_kernel.lengthscales = 3.0
    layer2.kern.index_kernel.variance = 20.0
    layer2.kern.basekern.lengthscales = 1.5
    layer2.kern.basekern.variance = 20.0

    like = MoBernoulli()

    return gpflux.DeepGP(np.zeros([X.shape[0], 0]), X,
                         layers=[layer0, layer1, layer2],
                         likelihood=like,
                         batch_size=minibatch_size,
                         name="my_latent_deep_gp")

@ex.capture
def setup_monitor_tasks(Xs, model, optimizer,
                        hz, hz_slow, basepath, adam_lr):

    tb_path = os.path.join(basepath, NAME, "tensorboards", experiment_name())
    model_path = os.path.join(basepath, NAME, experiment_name())
    fw = mon.LogdirWriter(tb_path)

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

    error_fast = lambda *args, **kwargs: calculate_nll(model, Xs[:200])
    error_slow = lambda *args, **kwargs: calculate_nll(model, Xs[:5000])

    tasks += [\
          mon.ScalarFuncToTensorBoardTask(fw, error_fast, "error")\
          .with_name('error')\
          .with_condition(mon.PeriodicIterationCondition(hz))\
          .with_exit_condition(True)\
          .with_flush_immediately(True)]

    tasks += [\
          mon.ScalarFuncToTensorBoardTask(fw, error_slow, "error_full")\
          .with_name('error_full')\
          .with_condition(mon.PeriodicIterationCondition(hz_slow))\
          .with_exit_condition(True)\
          .with_flush_immediately(True)]

    return tasks


@ex.capture
def setup_optimizer(model, global_step, adam_lr):

    if adam_lr == "decay":
        print("decaying lr")
        lr = 0.01 * 1.0 / (1 + global_step // 5000 / 3)
        # lr = tf.train.exponential_decay(learning_rate=0.01, global_step=global_step, 
        # decay_steps=-10000, decay_rate=10, staircase=True)
    else:
        lr = adam_lr

    return gpflow.train.AdamOptimizer(lr)

@ex.capture
def run(model, session, global_step, monitor_tasks, optimizer, iterations):

    monitor = mon.Monitor(monitor_tasks, session, global_step, print_summary=True)

    with monitor:
        optimizer.minimize(model, step_callback=monitor, maxiter=iterations, global_step=global_step)
    return model.compute_log_likelihood()


@ex.capture
def finish(X, Xs, model, basepath):

    nll = calculate_nll(model, Xs[:5000])
    print("error test", nll)

    fn = os.path.join(basepath, NAME, experiment_name() + ".gpflow")
    gpflow.Saver().save(fn, model)
    print("model saved")

    fn = os.path.join(basepath, NAME, experiment_name() + ".nll")
    np.savetxt(fn, nll)


def init_pca_matrix(X):

    X = (X - np.mean(X, axis=0))
    # X /= (np.std(X, axis=0) + 1e-6)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, whiten=True)
    T = pca.fit_transform(X)
    A = pca.components_.T / np.sqrt(5)  # P x L
    return A



    print(np.std(T2, axis=0))

    print(T.shape)
    print(T2.shape)
    # print(T/T2)
    # np.testing.assert_array_almost_equal(T, T2)


@ex.automain
def main():

    X, Xs = data()

    A = init_pca_matrix(X[:100])
    X = (X - np.mean(X, axis=0))
    T = (X @ A)

    import matplotlib.pyplot as plt
    plt.plot(T[:, 0], T[:, 1], "ko", alpha=.1)
    plt.show()

    return 0


    model = setup_model(X)

    model.compile()
    sess = model.enquire_session()
    step = mon.create_global_step(sess)

    # restore_session(sess)
    print("X", np.min(X), np.max(X))
    print("Xs", np.min(Xs), np.max(Xs))

    print("before optimisation ll", model.compute_log_likelihood())

    optimizer = setup_optimizer(model, step)
    monitor_tasks = setup_monitor_tasks(Xs, model, optimizer)
    ll = run(model, sess, step,  monitor_tasks, optimizer)
    print("after optimisation ll", ll)

    finish(X, Xs, model)

    return SUCCESS
