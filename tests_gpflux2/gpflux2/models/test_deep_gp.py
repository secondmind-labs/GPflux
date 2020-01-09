import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

import tensorflow as tf
import tqdm

from gpflow.kernels import RBF, Matern12
from gpflow.likelihoods import Gaussian
from gpflow.optimizers import Scipy
from gpflow.mean_functions import Identity, Zero
from gpflow.utilities import deepcopy_components

from gpflux2.models import DeepGP
from gpflux2.layers import GPLayer, LikelihoodLayer
from gpflux2.helpers import construct_basic_kernel, construct_basic_inducing_variables

# TODO: This file currently actually only really demos how to build and fit a DeepGP.
# Tests will be required.


def build_deep_gp(input_dim):
    layers = [input_dim, 2, 2, 1]
    # Below are different ways to build layers

    # 1. Pass in Lists:
    kernel_list = [RBF(), Matern12()]
    num_inducing = [25, 25]
    l1_kernel = construct_basic_kernel(kernels=kernel_list)
    l1_inducing = construct_basic_inducing_variables(
        num_inducing=num_inducing, input_dim=layers[0]
    )

    # 2. Pass in kernels, specificy output dims (shared hyperparams/variables)
    l2_kernel = construct_basic_kernel(
        kernels=RBF(), output_dim=layers[2], share_hyperparams=True
    )
    l2_inducing = construct_basic_inducing_variables(
        num_inducing=25, input_dim=layers[1], share_variables=True
    )

    # 3. Pass in kernels, specificy output dims (independent hyperparams/vars)
    # By default and the constructor will make indep. copies
    l3_kernel = construct_basic_kernel(kernels=RBF(), output_dim=layers[3])
    l3_inducing = construct_basic_inducing_variables(
        num_inducing=25, input_dim=layers[2], output_dim=layers[3]
    )

    # Assemble at the end
    gp_layers = [
        GPLayer(l1_kernel, l1_inducing),
        GPLayer(l2_kernel, l2_inducing),
        GPLayer(l3_kernel, l3_inducing, mean_function=Zero(), use_samples=False),
    ]
    return DeepGP(gp_layers, likelihood_layer=LikelihoodLayer(Gaussian()))


def train_deep_gp():
    tf.keras.backend.set_floatx("float64")
    input_dim = 2
    X, Y = setup_dataset(input_dim, 1000)
    deep_gp = build_deep_gp(input_dim)

    test = deep_gp(X)
    fig, plotter = get_live_plotter((X, Y), deep_gp)

    optimizer = tf.keras.optimizers.Adam()

    adam = tf.optimizers.Adam()

    calculate_objective = lambda: -deep_gp.elbo(X, Y)

    @tf.function(autograph=False)
    def step():
        adam.minimize(calculate_objective, deep_gp.trainable_weights)

    tq = tqdm.tqdm(range(int(80e3)))
    for i in tq:
        step()
        if i % 60 == 0:
            tq.set_postfix_str(f"objective: {calculate_objective()}")
            plotter()


def setup_dataset(input_dim: int, num_data: int):
    lim = [0, 100]
    kernel = RBF(lengthscale=20)
    sigma = 0.01
    X = np.random.random(size=(num_data, input_dim)) * lim[1]
    cov = kernel.K(X) + np.eye(num_data) * sigma ** 2
    Y = np.random.multivariate_normal(np.zeros(num_data), cov)[:, None]
    Y = np.clip(Y, -0.5, 0.5)
    return X, Y


def plot(train_data, mean=None):
    fig = plt.figure()
    plt.scatter(*train_data)
    plt.show()


def get_live_plotter(train_data, model):
    plt.ion()

    X, Y = train_data
    fig = plt.figure()

    ax = plt.axes(projection="3d")
    ax.set_zlim3d(2 * np.min(Y), 2 * np.max(Y))
    ax.set_xlim3d(np.min(X[:, 0]), np.max(X[:, 0]))
    ax.set_ylim3d(np.min(X[:, 1]), np.max(X[:, 1]))

    line = ax.scatter3D(X[:, 0], X[:, 1], Y, s=2, c="#003366")

    test_xx = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 50)
    test_yy = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 50)

    XX, YY = np.meshgrid(test_xx, test_yy)
    sample_points = np.hstack([XX.reshape(-1, 1), YY.reshape(-1, 1)])
    ZZ = np.zeros_like(XX)

    contour_line = ax.plot_wireframe(XX, YY, ZZ, linewidth=0.5)

    def plotter(*args, **kwargs):
        nonlocal contour_line

        ZZ_val = model(sample_points)
        if isinstance(ZZ_val, tuple):
            ZZ_val = ZZ_val[0]
        ZZ_hat = ZZ_val.numpy().reshape(XX.shape)
        ax.collections.remove(contour_line)
        contour_line = ax.plot_wireframe(XX, YY, ZZ_hat, linewidth=0.5)
        plt.draw()
        plt.pause(0.0001)

    return fig, plotter


if __name__ == "__main__":
    data = setup_dataset(2, 500)

    train_deep_gp()
    input()
