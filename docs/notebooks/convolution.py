# %% [markdown]
# ## 

# In this notebook we revisit the Convolutional Gaussian Processes (ConvGP), <cite data-cite="van2017convolutional"/>. Similarly to convolutional neural networks, ConvGP suits very well to model image processing tasks. As well as CNN, the ConvGP is endowed with translation invariant property. ConvGP imposes stronger and structured prior on a image response function $f(\cdot)$, using patch response function $g(\cdot) \sim GP(0, k_g(\cdot, \cdot))$. The image response function is a sum of patch responses for all (overlapping) patches in the image $f(\mathbb{x}) = \sum_{p=1}^{P}g(\mathbb{x}^{[p]})$, where $p$ is an index of a patch of image, and therefore $f(\cdot) \sim GP(0, \sum_p \sum_{p'} k_g(x^{[p]}, x^{[p']}))$. In a way, the patch response kernel can be viewed as an equivalent to a convolutional kernel of CNN.
#
# <img src="./convgp.png" alt="convgp" width="400px"/>

# %% [markdown]
# In this demo we consider a toy dataset: rectangle binary classification. We generate images with non-filled rectangles, and assign `0` class (label) to images with wide rectangles (a rectangle that has long horizontal side and short vertical side), and `1` class otherwise.

# %% [markdown]
# First, we import required components from :mod:`~gpflux` and :mod:`~gpflow`, and define constants for the demo:

# %%
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

import gpflow
import gpflux
import tensorflow as tf

from gpflux.layers import LikelihoodLayer, GPLayer
from gpflux.helpers import construct_basic_kernel
from gpflow.kernels import Convolutional, SquaredExponential
from gpflow.inducing_variables import InducingPatches, SharedIndependentInducingVariables

Shape = Tuple[int, int]
Data = Tuple[tf.Tensor, tf.Tensor]

np.random.seed(123)
tf.random.set_seed(42)

#%% [markdown]
# Constants:

# %%
maxiter = 100
num_train_data = 100
batch_size = 25
num_inducing = 20
num_test_data = 300
num_epochs = 100
height = width = 14
image_shape = (height, width)
patch_shape = (3, 3)

# %% [markdown]
# Dataset generation code ([from GPflow convolutions notebook](https://gpflow.readthedocs.io/en/master/notebooks/advanced/convolutional.html?highlight=make_rectangle), :cite:p:`gpflow2020`):

# %%
def make_rectangle(arr, x0, y0, x1, y1):
    arr[y0:y1, x0] = 1
    arr[y0:y1, x1] = 1
    arr[y0, x0:x1] = 1
    arr[y1, x0 : x1 + 1] = 1


def make_random_rectangle(arr):
    x0 = np.random.randint(1, arr.shape[1] - 3)
    y0 = np.random.randint(1, arr.shape[0] - 3)
    x1 = np.random.randint(x0 + 2, arr.shape[1] - 1)
    y1 = np.random.randint(y0 + 2, arr.shape[0] - 1)
    make_rectangle(arr, x0, y0, x1, y1)
    return x0, y0, x1, y1


def make_rectangles_dataset(num, w, h):
    d, Y = np.zeros((num, h, w)), np.zeros((num, 1))
    for i, img in enumerate(d):
        for j in range(1000):  # Finite number of tries
            x0, y0, x1, y1 = make_random_rectangle(img)
            rw, rh = y1 - y0, x1 - x0
            if rw == rh:
                img[:, :] = 0
                continue
            Y[i, 0] = rw > rh
            break
    return (
        d.reshape(num, w * h).astype(gpflow.config.default_float()),
        Y.astype(gpflow.config.default_float()),
    )


data = make_rectangles_dataset(num_train_data, *image_shape)
test_data = make_rectangles_dataset(num_test_data, *image_shape)
x, y = data
xt, yt = test_data

# %% [markdown]
# Examples from rectangle dataset:

# %%
plt.figure(figsize=(8, 3))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(x[i, :].reshape(*image_shape))
    plt.title(f"Class = {int(y[i, 0])}")
    plt.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        left=False,
        labelleft=False,
        bottom=False,
        top=False,
    )

plt.show()

# %% [markdown]
# In the following steps we create a GP convolutional layer and a Bernoulli likelihood layer:
# 1. Create a convolutional kernel using :mod:`~gpflow.kernels.Convolutional`, and specify input's image shape and a patch shape.
# 2. Randomly select image patches to initialize inducing points :mod:`~gpflow.inducing_variables.SharedIndependentInducingVariables`.
# 3. And, finally we construct convolutional gaussian process and likelihood layers.

# %%
patch_size = np.prod(patch_shape)
x, y = data
num_data = x.shape[0]
kernel_conv = Convolutional(SquaredExponential(), image_shape, patch_shape)
patches = kernel_conv.get_patches(x).numpy()
patches = np.unique(patches.reshape(-1, patch_size), axis=0)[:num_inducing]
inducing_patches = InducingPatches(patches)

kernel_layer = construct_basic_kernel(kernel_conv, output_dim=1, share_hyperparams=True)
inducing_layer = SharedIndependentInducingVariables(inducing_patches)

convgp_layer = GPLayer(
    kernel=kernel_layer,
    inducing_variable=inducing_layer,
    num_data=num_data,
    num_latent_gps=1,
    mean_function=gpflow.mean_functions.Zero(),
    name="gplayer",
)

likelihood = gpflow.likelihoods.Bernoulli()
likelihood_layer = LikelihoodLayer(likelihood)

# %% [markdown]
# Below are are going to use Keras for training convolutional GP model. The details of GPflux and Keras itegration you can find [here](HOW TO ADD A CROSS REFERENCE TO A FILE?).

#%%
conv_model = gpflux.models.DeepGP(
    [convgp_layer], likelihood_layer, default_model_class=tf.keras.Model
)

#%% We use keras Callbacks to control the learning rate and monitor the convergence status of the loss function (ELBO).
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss",
        patience=5,
        factor=0.95,
        verbose=1,
        min_lr=1e-5,
    )
]

conv_train = conv_model.as_training_model()
conv_train.compile(tf.optimizers.Adam(learning_rate=0.1))

history = conv_train.fit(
    {"inputs": x, "targets": y}, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks
)

conv_model_test = conv_model.as_prediction_model()
predict_result = conv_model_test(xt)
predict_yt = np.array(predict_result.y_mean.numpy() > 0.5, dtype=int)


#%% [markdown]
# Classification error after training:

error = np.sum(predict_yt != yt) / yt.shape[0]
error


#%% [markdown]
# In this notebook we showed how to use GPflux to build and train shallow convolutional Gaussian processes using Keras. More advanced models like deep convolutional GPs require multi-output convoluitonal kernels and extensions in dispatchers of :mod:`~gpflow.conditionals` and in :mod:`~gpflow.covariances`.