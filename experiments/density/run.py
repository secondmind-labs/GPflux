from typing import List, Optional

import os

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sklearn.decomposition import PCA

import gpflow
import gpflow.training.monitor as mon
import gpflux
from data import fixed_binarized_mnist
from utils import calculate_nll, plot_latents, plot_samples, plot_inducing_patches

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

class PcaResnetEncoder(gpflux.encoders.RecognitionNetwork):

    def __init__(self,
                 latent_dim: int,
                 mean: np.ndarray,
                 pca_projection_matrix: np.ndarray,
                 network_dims: List[int],
                 activation_func = None,
                 name: Optional[str] = None):
        input_dim = pca_projection_matrix.shape[1]
        super().__init__(latent_dim, input_dim, network_dims, activation_func=activation_func, name=name)
        self.pca_projection_matrix = pca_projection_matrix  # [D x L]
        self.mean = mean
        for w in self.Ws:
            w = w.read_value() / 5.0  # reduce weights

    @gpflow.decors.params_as_tensors
    def __call__(self, Z: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        Z = tf.matmul(Z - self.mean, self.pca_projection_matrix)  # [N, L]
        m, v = super().__call__(Z)
        return m + Z, tf.nn.softplus(v - 1.0)
    
    @gpflow.decors.autoflow((gpflow.settings.float_type, [None, None]))
    def eval(self, X):
        return self.__call__(X)
    

def init_pca_projection_matrix(X, latent_dim, norm=True):
    pca = PCA(n_components=latent_dim, whiten=True).fit(X)
    div = np.sqrt(5) if norm else 1.0
    return pca.components_.T / div  # P x L


@ex.config
def config():

    # number of inducing points
    num_inducing_1 = 40
    # number of inducing points
    num_inducing_2 = 750
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

    # path of model that should be restored
    restore = ""

    # latent dim
    latent_dim = 2

    # number of gps in the first layer
    num_gps1 = 10

    # print hz
    hz = 20
    hz_slow = 500



@ex.capture
def data(basepath):

    path = os.path.join(basepath, "data")
    X, Xs = fixed_binarized_mnist(path)

    return X, Xs

def _strip_restore(restore):
    if restore != "":
        if restore.endswith("/"):
            restore = restore.split("/")[-2]
        else:
            restore = restore.split("/")[-1]
    return restore


@ex.capture
def experiment_name(adam_lr, num_inducing_1, num_inducing_2, minibatch_size, patch_size, latent_dim, num_gps1,
                    restore):
    if restore != "":
        restore = _strip_restore(restore)
        return restore + "_restored"

    args = np.array(
        [
            "mean",
            "matern",
            "adam", adam_lr,
            "M1", num_inducing_1,
            "M2", num_inducing_2,
            "gps1", num_gps1,
            "L", latent_dim,
            "minibatch_size", minibatch_size,
            "patch", patch_size[0],
        ])
    return "_".join(args.astype(str))


@ex.capture
def restore_session(session, restore, basepath):
    if restore == "":
        return

    restore = _strip_restore(restore)
    model_path = os.path.join(basepath, NAME, restore)
    print(model_path)
    mon.restore_session(session, model_path)
    print("Model restored")


@gpflow.defer_build()
@ex.capture
def setup_model(X, minibatch_size, patch_size, num_inducing_1, num_inducing_2, latent_dim, num_gps1):

    assert patch_size[0] == patch_size[1]
    assert patch_size[0] == 5

    # H = int(X.shape[1]**.5)

    ## layer 0
    # enc = gpflux.encoders.RecognitionNetwork(latent_dim, X.shape[1], [200, 200, 100])
    A = init_pca_projection_matrix(X[:1000], latent_dim)
    enc = PcaResnetEncoder(latent_dim, np.mean(X, axis=0), A, network_dims=[5, 5])
    layer0 = gpflux.layers.LatentVariableConcatLayer(latent_dim, encoder=enc)

    ## layer 1
    X_tmp = np.zeros([1000, 32, 32])
    X_tmp[:, 2:30, 2:30] = X[:1000, :].reshape(-1, 28, 28)
    X_tmp = X_tmp.reshape(-1, 32**2)
    PCA = init_pca_projection_matrix(X_tmp, latent_dim, norm=False)  # P x L
    mean = gpflow.mean_functions.Linear(A=PCA.T)

    W = np.random.randn(32**2, num_gps1) / num_gps1
    kern = gpflow.multioutput.kernels.SharedMixedMok(
                gpflow.kernels.Matern52(latent_dim), W)
    feat = gpflow.multioutput.features.MixedKernelSharedMof(
                gpflow.features.InducingPoints(np.random.randn(num_inducing_1, latent_dim)))
    layer1 = gpflux.layers.GPLayer(kern, feat, num_latents=num_gps1, mean_function=mean)
    # layer1.kern.kern.lengthscales = 0.3

    ## layer 2: indexed conv patch 5x5
    Y1 = np.random.randn(100, latent_dim) @ PCA.T
    Y1 = (np.reshape(Y1, (-1, 32, 32)))[:, 8:24, 8:24]
    init_patches = gpflux.init.PatchSamplerInitializer(Y1, width=16, height=16, unique=True)
    patches = init_patches.sample([num_inducing_2, *patch_size])

    proj = patches
    width = 5
    vmin, vmax = proj.min(), proj.max()

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for patch, ax in zip(proj, axes.flat):
        im = ax.matshow(patch.reshape(width, width), vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_ax = fig.add_axes([0.95, .1, .01, 0.8])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()
    exit(0)

    # patches = ((patches - np.mean(patches, axis=0)) / (np.std(patches, axis=0) + 1e-6)) / 3.0
    # patches = np.random.randn(*patches.shape) * 1e-3

    layer2 = gpflux.layers.ConvLayer([32, 32], [28, 28], num_inducing_2, 
                                     patch_size=[5, 5], 
                                     with_indexing=True,
                                     patches_initializer=init_patches)

    # layer2.q_sqrt = layer2.q_sqrt.read_value() * 1e-3
    # layer2.q_sqrt.set_trainable(True)
    # layer2.feature.set_trainable(True)


    like = MoBernoulli()

    model = gpflux.DeepGP(np.zeros([X.shape[0], 0]), X,
                         layers=[layer0, layer1, layer2],
                         likelihood=like,
                         batch_size=minibatch_size,
                         name="my_latent_deep_gp")

    ## RE-INIT SOME PARAMS
    model.layers[1].q_sqrt = layer1.q_sqrt.read_value() * 1e-3
    # layer2.kern.index_kernel.lengthscales = 3.0
    # layer2.kern.index_kernel.variance = 1.0
    # layer2.kern.basekern.variance = 1.0
    model.layers[2].kern.basekern.lengthscales = 0.02


    ## FIX SOME PARAMS
    # model.layers[0].encoder.set_trainable(False)
    model.layers[1].mean_function.set_trainable(False)
    model.layers[1].feature.set_trainable(False)
    model.layers[2].feature.set_trainable(False)
    # layer1.q_sqrt.set_trainable(True)
    # layer2.kern.basekern.lengthscales.set_trainable(True)
    
    model.alpha = 1.0

    print(model.as_pandas_table())
    return model

@ex.capture
def setup_monitor_tasks(Xs, model, optimizer, latent_dim,
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
        .with_condition(mon.PeriodicIterationCondition(hz))]

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

    kl_global_func = lambda *args, **kwargs: model.compute_KL_global()
    kl_minibatch_func = lambda *args, **kwargs: model.compute_KL_minibatch()

    tasks += [\
          mon.ScalarFuncToTensorBoardTask(fw, kl_global_func, "kl_global")\
          .with_name('kl_global')\
          .with_condition(mon.PeriodicIterationCondition(hz))\
          .with_exit_condition(True)\
          .with_flush_immediately(True)]

    tasks += [\
          mon.ScalarFuncToTensorBoardTask(fw, kl_minibatch_func, "kl_minibatch")\
          .with_name('kl_minibatch')\
          .with_condition(mon.PeriodicIterationCondition(hz))\
          .with_exit_condition(True)\
          .with_flush_immediately(True)]

    plot_patches_func = plot_inducing_patches(model)
    tasks += [\
        mon.ImageToTensorBoardTask(fw, plot_patches_func, "plot_patches")\
        .with_name('plot_patches')\
        .with_condition(mon.PeriodicIterationCondition(hz))\
        .with_exit_condition(True)\
        .with_flush_immediately(True)]

    plot_samples_func = plot_samples(model)
    tasks += [\
        mon.ImageToTensorBoardTask(fw, plot_samples_func, "plot_samples")\
        .with_name('plot_samples')\
        .with_condition(mon.PeriodicIterationCondition(hz))\
        .with_exit_condition(True)\
        .with_flush_immediately(True)]

    if latent_dim == 2:
        plot_latents_func = plot_latents(model, Xs)
        tasks += [\
            mon.ImageToTensorBoardTask(fw, plot_latents_func, "plot_latents")\
            .with_name('plot_latents')\
            .with_condition(mon.PeriodicIterationCondition(hz))\
            .with_exit_condition(True)\
            .with_flush_immediately(True)]

    # /////////// SLOW TASKS, only ran very occasionally... //////////////

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


# def run_nat_grad(model, session, global_step, monitor_tasks, optimizer, iterations):
#     try:
#         with mon.Monitor(monitor_tasks, sess, global_step, print_summary=True) as monitor:
#             try:
#                 mon.restore_session(sess, checkpoint_path)
#             except ValueError:
#                 pass

#             print(sess.run(global_step))

#             for it in range(max([ARGS.iterations - sess.run(global_step), 0])):
#                 sess.run(op_increment)
#                 monitor()
#                 sess.run(op_ng)
#                 sess.run(op_adam)

#     except(KeyboardInterrupt):
#         pass

@ex.capture
def run(model, session, global_step, monitor_tasks, optimizer, iterations):

    monitor = mon.Monitor(monitor_tasks, session, global_step, print_summary=True)

    with monitor:
        optimizer.minimize(model, step_callback=monitor, maxiter=iterations, global_step=global_step)
        # model.layers[0].encoder.set_trainable(True)
        # model.layers[1].q_sqrt.set_trainable(True)
        # model.layers[2].q_sqrt.set_trainable(True)
        # model.alpha = 1.0
        # optimizer.minimize(model, step_callback=monitor, maxiter=iterations, global_step=global_step)
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


from sklearn.decomposition import PCA

class MyPca:
    def __init__(self, X, L):
        pca = PCA(n_components=L, whiten=True)
        pca.fit(X)
        self.mean = pca.mean_
        C = pca.components_.T  # D x L
        v = np.sqrt(pca.explained_variance_)[None, :]  # 1 x L
        self.A = C / v
        self.B = C * v
    
    def down(self, X):
        X = X - self.mean
        X = np.matmul(X, self.A)
        return X
    
    def up(self, X):
        return np.matmul(X, self.B.T) + self.mean


@ex.automain
def main():

    X, Xs = data()

    pca = MyPca(X[:500], 100)

    XX = X[500:510]
    low = pca.down(XX)
    XXr = pca.up(low)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(10, 2)
    for i, (x, xr) in enumerate(zip(XX, XXr)):
        axes[i, 0].matshow(x.reshape(28, 28))
        axes[i, 1].matshow(xr.reshape(28, 28))
    plt.show()
    

    return 0
    
    L = 3
    X = X[:100]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=L, whiten=True)
    Y = pca.fit_transform(X)
    X2 = pca.inverse_transform(Y)
    print(X.shape)
    print(Y.shape)

    print(X - X2)
    print(np.mean((X - X2)**2))

    return 0



    # L = 10
    # PCA = init_pca_projection_matrix(X, L).T  # L x P

    # # # ww = np.linspace(-2, 2, 5)
    # # # Ws = np.vstack([x.flatten() for x in np.meshgrid(ww, ww)]).T  # Nx2
    # Ws = np.random.randn(100, L)

    # proj = Ws @ PCA

    # vmin, vmax = proj.min(), proj.max()

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    # for patch, ax in zip(proj, axes.flat):
    #     im = ax.matshow(patch.reshape(28, 28), vmin=vmin, vmax=vmax)
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    # cbar_ax = fig.add_axes([0.95, .1, .01, 0.8])
    # fig.colorbar(im, cax=cbar_ax)
    # plt.show()

    # return 0

    model = setup_model(X)
    model.compile()

    sess = model.enquire_session()
    step = mon.create_global_step(sess)

    restore_session(sess)
    print("X", np.min(X), np.max(X))
    print("Xs", np.min(Xs), np.max(Xs))

    print("before optimisation ll", model.compute_log_likelihood())

    optimizer = setup_optimizer(model, step)
    monitor_tasks = setup_monitor_tasks(Xs, model, optimizer)
    ll = run(model, sess, step,  monitor_tasks, optimizer)
    print("after optimisation ll", ll)

    finish(X, Xs, model)

    return SUCCESS
