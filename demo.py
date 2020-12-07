import sys

import matplotlib
import numpy as np
import tensorflow as tf
import tqdm
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler

from gpflow.kernels import RBF, Matern12
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Zero

from gpflux.helpers import construct_basic_inducing_variables, construct_basic_kernel
from gpflux.layers import GPLayer, LikelihoodLayer
from gpflux.models import DeepGP
from gpflux.utils.live_plotter import live_plot

sys.path.append("automl_implementations")
from dgp_2017 import DoublyStochasticDGP2017

tf.keras.backend.set_floatx("float64")
matplotlib.use('TkAgg')


def setup_dataset(input_dim: int, num_data: int):
    lim = [0, 100]
    kernel = RBF(lengthscales=20)
    sigma = 0.01
    X = np.random.random(size=(num_data, input_dim)) * lim[1]
    cov = kernel.K(X) + np.eye(num_data) * sigma ** 2
    Y = np.random.multivariate_normal(np.zeros(num_data), cov)[:, None]
    Y = np.clip(Y, -0.5, 0.5)
    return X, Y


_ = mplot3d  # importing mplot3d has side-effects; we don't actually need to use it


def get_live_plotter(model, X, Y):
    assert matplotlib.get_backend() == 'TkAgg'

    fig = plt.figure()
    plt.ion()
    plt.show()

    ax = plt.axes(projection="3d")

    ax.set_zlim3d(2 * np.min(Y), 2 * np.max(Y))
    ax.set_xlim3d(np.min(X[:, 0]), np.max(X[:, 0]))
    ax.set_ylim3d(np.min(X[:, 1]), np.max(X[:, 1]))

    _ = ax.scatter3D(X[:, 0], X[:, 1], Y, s=2, c="#003366")

    test_xx = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 50)
    test_yy = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 50)

    XX, YY = np.meshgrid(test_xx, test_yy)
    sample_points = np.hstack([XX.reshape(-1, 1), YY.reshape(-1, 1)])
    ZZ = np.zeros_like(XX)

    contour_line = ax.plot_wireframe(XX, YY, ZZ, linewidth=0.5)

    @live_plot
    def plot(model, fig=None, axes=None):
        nonlocal contour_line

        ZZ_val = model((sample_points, None), training=False)
        if isinstance(ZZ_val, tuple):
            ZZ_val = ZZ_val[0]

        ZZ_hat = ZZ_val.numpy().reshape(XX.shape)
        ax.collections.remove(contour_line)
        contour_line = ax.plot_wireframe(XX, YY, ZZ_hat, linewidth=0.5)

    def plotter(model, X, Y):
        plot(model, fig=fig, axes=[])

    return fig, plotter


def main():
    X, Y = setup_dataset(2, 1000)
    transformer_X = StandardScaler().fit(X)
    transformer_Y = StandardScaler().fit(Y)
    tX = transformer_X.transform(X)
    tY = transformer_Y.transform(Y)
    automl_model = DoublyStochasticDGP2017(tX, tY, optimize=False)
    keras_model = automl_model.model
    _, plot = get_live_plotter(keras_model, tX, tY[:, 0])

    class CB(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epochs, *args):
            if epochs % 100 == 0:
                plot(keras_model, tX, tY)

    automl_model._optimize(tX, tY, callbacks=[CB()])


if __name__ == "__main__":
    main()
