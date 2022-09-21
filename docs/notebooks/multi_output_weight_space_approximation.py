# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Weight Space Approximation with Random Fourier Features
"""

# %% [markdown]
r"""
This notebook demonstrates how to approximate an exact Gaussian process regression model (GPR) with random Fourier features in weight space. The end result is Figure 1 from from Wilson et al. "Efficiently sampling functions from Gaussian process posteriors" <cite data-cite="wilson2020efficiently"/>. This figure demonstrates that approximating an exact GPR model in weight space becomes increasingly more difficult as the training data grows. While the approximation remains accurate in areas within the training data and far away from the training data, predictions close to the training data points (but outside the training data interval) become less reliable. Note that Wilson et al. provide a method to alleviate exactly this problem; however, this is outside the scope of this notebook, where the emphasis is on how to build a weight space approximated Gaussian process model with random Fourier features in `gpflux`.

The basic idea is to approximate a stationary kernel $k(X,X^\prime)$ for one-dimensional inputs $X \in \mathbb{R}$ and $X^\prime \in \mathbb{R}$ according to Bochner's theorem:

$$
k(X, X^\prime) \approx \sum_{i=1}^I \phi_i(X) \phi_i(X^\prime),
$$
with $I$ Fourier features $\phi_i$  following Rahimi and Recht "Random features for large-scale kernel machines" (NeurIPS, 2007) defined as

$$
\phi_i(X) = \sqrt{\frac{2 \sigma^2}{l}} \cos(\theta_i X + \tau_i),
$$
where $\sigma^2$ refers to the kernel variance and $l$ to the kernel lengthscale. $\theta_i$ and $\tau_i$ are randomly drawn hyperparameters that determine each feature function $\phi_i$. The hyperparameter $\theta_i$ is randomly drawn from the kernel's spectral density. The spectral density of a stationary kernel is obtained by interpreting the kernel as a function of one argument only (i.e. the distance between $X$ and $X^\prime$) and performing a Fourier transform on that function, resulting in an unnormalised probability density (from which samples can be obtained). The hyperparameter $\tau_i$ is obtained by sampling from a uniform distribution $\tau_i \sim \mathcal{U}(0,2\pi)$. Note that both $\theta_i$ and $\tau_i$ are fixed and not optimised over. An interesting direction of future research is how to automatically identify those (but this is outside the scope of this notebook). If we drew infinitely many samples, i.e. $I \rightarrow \infty$, we would recover the true kernel perfectly.

The kernel approximation specified above enables you to express a supervised inference problem with training data $\mathcal{D} = \{(X_n,y_n)\}_{n=1,...,N}$ in weight space view as

$$
p(\textbf{w} | \mathcal{D}) = \frac{\prod_{n=1}^N p(y_n| \textbf{w}^\intercal \boldsymbol{\phi}(X_n), \sigma_\epsilon^2) p(\textbf{w})}{p(\mathcal{D})},
$$
where we assume $p(\textbf{w})$ to be a standard normal multivariate prior and $p(y_n| \textbf{w}^\intercal \boldsymbol{\phi}(X_n), \sigma_\epsilon^2)$ to be a univariate Gaussian observation model of the i.i.d. likelihood with mean $\textbf{w}^\intercal \boldsymbol{\phi}(X_n)$ and noise variance $\sigma_\epsilon^2$. The boldface notation $\boldsymbol{\phi}(X_n)$ refers to the vector-valued feature function that evaluates all features from $1$ up to $I$ for one particular input $X_n$. Under these assumptions, the posterior $p(\textbf{w} | \mathcal{D})$ enjoys a closed form and is Gaussian. Predictions can readily be obtained by sampling $\textbf{w}$ and evaluating the function sample $\textbf{w}$ at new locations $\{X_{n^\star}^\star\}_{n^\star=1,...,N^\star}$ as $\{\textbf{w}^\intercal \boldsymbol{\phi}(X_{n^\star}^\star)\}_{n^\star=1,...,N^\star}$.

The advantage of expressing a Gaussian process in weight space is that functions are represented as weight vectors $\textbf{w}$ (rather than actual functions $f(\cdot)$) from which samples can be obtained a priori without knowing where the function should be evaluated. When expressing a Gaussian process in function space view the latter is not possible, i.e. a function $f(\cdot)$ cannot be sampled without knowing where to evaluate the function, namely at $\{X_{n^\star}^\star\}_{n^\star=1,...,N^\star}$. Weight space approximated Gaussian processes therefore hold the potential to sample efficiently from Gaussian process posteriors, which is desirable in vanilla supervised learning but also in domains such as Bayesian optimisation or model-based reinforcement learning.

In the following example, we compare a weight space approximated GPR model (WSA model) with both a proper GPR model and a sparse variational Gaussian Process model (SVGP). GPR models and SVGP models are implemented in `gpflow`, but the two necessary ingredients for building the WSA model are part of `gpflux`: these are random Fourier feature functions via the `RandomFourierFeaturesCosine` class, and approximate kernels based on Bochner's theorem (or any other theorem that approximates a kernel with a finite number of feature functions, e.g. Mercer) via the `KernelWithFeatureDecomposition` class.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20, 7)
plt.rc("text")
plt.rcParams.update({"font.size": 20})
import tensorflow as tf

import gpflow
from gpflow.config import default_float
from gpflow.models import GPR, SVGP
from gpflow.kernels import SquaredExponential, Matern52
from gpflow.likelihoods import Gaussian
from gpflow.inducing_variables import InducingPoints

from gpflux.layers.basis_functions.fourier_features import MultiOutputRandomFourierFeaturesCosine
from gpflux.feature_decomposition_kernels import (
    KernelWithFeatureDecomposition,
    SeparateMultiOutputKernelWithFeatureDecomposition,
    SharedMultiOutputKernelWithFeatureDecomposition,
)


# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import tensorflow as tf

import gpflow


# from gpflow import posteriors
from gpflow.base import InputData, MeanAndVariance, RegressionData
from gpflow.experimental.check_shapes import check_shapes, inherit_check_shapes
from gpflow.kernels import Kernel
from gpflow.likelihoods import Gaussian
from gpflow.logdensities import multivariate_normal
from gpflow.mean_functions import MeanFunction
from gpflow.utilities.model_utils import add_likelihood_noise_cov
from gpflow.utilities import assert_params_false
from gpflow.models.model import GPModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor


class GPR_deprecated(GPModel, InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this model is given by

    .. math::
       \log p(Y \,|\, \mathbf f) =
            \mathcal N(Y \,|\, 0, \sigma_n^2 \mathbf{I})

    To train the model, we maximise the log _marginal_ likelihood
    w.r.t. the likelihood variance and kernel hyperparameters theta.
    The marginal likelihood is found by integrating the likelihood
    over the prior, and has the form

    .. math::
       \log p(Y \,|\, \sigma_n, \theta) =
            \mathcal N(Y \,|\, 0, \mathbf{K} + \sigma_n^2 \mathbf{I})
    """

    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
    )
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: Optional[float] = None,
        likelihood: Optional[Gaussian] = None,
    ):
        assert (noise_variance is None) or (
            likelihood is None
        ), "Cannot set both `noise_variance` and `likelihood`."
        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.log_marginal_likelihood()

    @check_shapes(
        "return: []",
    )
    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        K = self.kernel(X)
        ks = add_likelihood_noise_cov(K, self.likelihood, tf.tile(X[None, ...], [2, 1, 1]))
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data
        points.
        """
        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        X, Y = self.data
        err = Y - self.mean_function(X)

        kmm = self.kernel(X)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X, Xnew)
        # kmm_plus_s = add_likelihood_noise_cov(kmm, self.likelihood, X)
        kmm_plus_s = add_likelihood_noise_cov(
            kmm, self.likelihood, tf.tile(X[None, ...], [2, 1, 1])
        )

        # NOTE -- this onlty works for a single latent Full GP
        # conditional = gpflow.conditionals.base_conditional

        conditional = gpflow.conditionals.util.separate_independent_conditional_implementation

        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var


# %% [markdown]
"""
Our aim is to demonstrate the decrease in predictive quality of a WSA model when increasing the number of training points. To that end, we perform two sets of experiments: one with few and one with many training data points. Each experiment compares a WSA model to an exact GPR and to an approximate SVGP model, resulting in six plots all in all.

We first define settings that remain the same across the two sets of experiments, like the interval of the training points, aspects of the generative model (i.e. kernel variance and lengthscale, and the variance of the observation model), and the number of feature functions of the WSA model.

The only aspect that is different across both experimental settings is the number of training data points. We increase the number of inducing points for the SVGP model to cope with this.
"""

# %%
# experiment parameters that are the same for both sets of experiments
X_interval = [0.14, 0.5]  # interval where training points live
lengthscale = [
    0.1
]  # lengthscale for the kernel (which is not learned in all experiments, the kernel variance is 1)
number_of_features = 2000  # number of basis functions for weight-space approximated kernels
noise_variance = 1e-3  # noise variance of the likelihood (which is not learned in all experiments)
number_of_test_samples = 1024  # number of evaluation points for prediction
number_of_function_samples = (
    20  # number of function samples to be drawn from (approximate) posteriors
)

# experiment parameters that differ across both sets of experiments
number_of_train_samples = [4, 1000]  # number of training points
number_of_inducing_points = [4, 8]  # number of inducing points for SVGP models

# kernel class
# kernel_class = Matern52  # set altern    experiment = 2*j + nvmatively kernel_class = RBF

# plotting configuration
x_lim = [0.0, 1.0]
y_lim = [-3.5, 3.5]

# %% [markdown]
"""
We proceed by generating the training data for both experimental settings from a ground truth function which is a sample from a prior zero-mean GP with a predefined kernel (in our case, we use a `Matern52` kernel but we could have chosen an `RBF` kernel -- both of which are defined in `gpflow`).
"""

# %%
# generate training data and evaluation points for both sets of experiments


list_kernels = [
    Matern52(lengthscales=lengthscale),
    SquaredExponential(lengthscales=lengthscale),
]
# kernel = kernel_class(lengthscales=lengthscale)  # kernel object to draw training dataset from

X, y, X_star = (
    [],
    [],
    [],
)  # training points, training observations, and test points for evaluation

# 1st iteration: experiments with few training points -- 2nd iteration: experiments with many training points

for i in range(len(number_of_train_samples)):

    X_temp, y_temp, X_star_temp = [], [], []

    # training pointsnumber_of_train_samples
    X.append(
        np.linspace(start=X_interval[0], stop=X_interval[1], num=number_of_train_samples[i])[
            ..., None
        ]
    )

    for j in range(len(list_kernels)):

        # training observations generated from a zero-mean GP corrupted with Gaussian noise
        kXX = list_kernels[j].K(X[-1])
        kXX_plus_noise_var = kXX + tf.eye(tf.shape(kXX)[0], dtype=kXX.dtype) * noise_variance
        lXX = tf.linalg.cholesky(kXX_plus_noise_var)
        y_temp.append(
            tf.matmul(lXX, tf.random.normal([number_of_train_samples[i], 1], dtype=X[-1].dtype),)[
                ..., 0
            ][..., None]
        )

    # test points for evaluation
    X_star.append(np.linspace(start=x_lim[0], stop=x_lim[1], num=number_of_test_samples)[..., None])
    y.append(np.concatenate(y_temp, axis=-1))


# %% [markdown]
"""
The `for` loop below iterates through both experimental settings with few and many training examples respectively. In each iteration, the GPR model is built first (and its prediction is used as "ground truth" to compare with the remaining models) followed by the SVGP model (which requires optimisation to identify internal parameters) and the WSA model.
"""

# %%
# create subplot frame
# 1st row: experiments with few training examples, 2nd row: experiments with many training examples :: Matern52 kernel
# 3rd row: experiments with few training examples, 4th row: experiments with many training examples :: SqExp kernel
# 1st col: exact Gaussian process regression (GPR), 2nd col: sparse variational Gaussian process model (SVGP),
# 3rd col: weight space approximation (WSA) of the exact GPR posterior with random Fourier features
fig, axs = plt.subplots(2, 2)


# 1st iteration: experiments with few training points -- 2nd iteration: experiments with many training points

for experiment in range(len(number_of_train_samples)):

    # subplot titles and axis labels
    axs[experiment, 0].set_title(
        "Weight Space GP Matern52 $N=" + str(number_of_train_samples[experiment]) + "$"
    )
    axs[experiment, 1].set_title(
        "Weight Space GP SqExp $N=" + str(number_of_train_samples[experiment]) + "$"
    )
    axs[experiment, 0].set_ylabel("$f(X)$")
    if experiment == 1:
        axs[experiment, 0].set_xlabel("$X$")
        axs[experiment, 1].set_xlabel("$X$")

    # plot training point locations X and set axis limits
    if (
        experiment == 0
    ):  # as vertical lines for the first set of experiments with few training samples
        axs[experiment, i].vlines(X[experiment], ymin=y_lim[0], ymax=y_lim[1], colors="lightgrey")
    else:  # as fill plots for the second set of experiments with many training samples
        axs[experiment, 1].fill_between(
            X[experiment].ravel(), y_lim[0], y_lim[1], color="gray", alpha=0.2
        )
    axs[experiment, 0].set_xlim(x_lim)
    axs[experiment, 0].set_ylim(y_lim)
    axs[experiment, 1].set_xlim(x_lim)
    axs[experiment, 1].set_ylim(y_lim)

    # create exact GPR model with weight-space approximated kernel (WSA model)

    kernel1 = gpflow.kernels.Matern52(lengthscales=lengthscale)
    kernel2 = gpflow.kernels.SquaredExponential(lengthscales=lengthscale)
    # kernel = gpflow.kernels.SeparateIndependent( kernels = [kernel1, kernel2])
    kernel = gpflow.kernels.SharedIndependent(kernel=kernel1, output_dim=2)

    feature_functions = MultiOutputRandomFourierFeaturesCosine(
        kernel, number_of_features, dtype=default_float()
    )

    feature_coefficients = np.ones((2, number_of_features, 1), dtype=default_float())
    # kernel = SeparateMultiOutputKernelWithFeatureDecomposition(
    #    kernel=None, feature_functions=feature_functions, feature_coefficients=feature_coefficients,
    #    output_dim = 2
    # )
    kernel = SharedMultiOutputKernelWithFeatureDecomposition(
        kernel=None,
        feature_functions=feature_functions,
        feature_coefficients=feature_coefficients,
        output_dim=2,
    )

    print("***************************************")
    print("-- shape of data for current experiment")
    print(X[experiment].shape)
    print(y[experiment].shape)
    print(X_star[experiment].shape)

    gpr_model = GPR_deprecated(
        data=(X[experiment], y[experiment]),
        kernel=kernel,
        noise_variance=noise_variance,
    )

    # predict function mean and variance, and draw function samples (without observation noise)#

    f_mean, f_var = gpr_model.predict_f(X_star[experiment])
    f_samples = gpr_model.predict_f_samples(
        X_star[experiment], num_samples=number_of_function_samples
    )
    f_mean_plus_2std = f_mean + 2 * f_var ** 0.5
    f_mean_minus_2std = f_mean - 2 * f_var ** 0.5

    print("***************************************")
    print("-- shape of current predictions")
    print(f_mean.shape)
    print(f_mean_minus_2std.shape)

    # visualise WSA model predictions (mean +/- 2 * std and function samples) in the third column

    ### Matern52 ###

    axs[experiment, 0].fill_between(
        X_star[experiment][..., 0],
        f_mean_minus_2std[..., 0],
        f_mean_plus_2std[..., 0],
        color="orange",
        alpha=0.2,
    )
    for i in range(f_samples.shape[0]):
        axs[experiment, 0].plot(
            X_star[experiment][..., 0],
            f_samples[i, ..., 0],
            color="orange",
            linewidth=0.2,
        )
    axs[experiment, 0].plot(X_star[experiment][..., 0], f_mean[..., 0], color="orange")

    ### SquaredExponential ###

    axs[experiment, 1].fill_between(
        X_star[experiment][..., 0],
        f_mean_minus_2std[..., 1],
        f_mean_plus_2std[..., 1],
        color="orange",
        alpha=0.2,
    )
    for i in range(f_samples.shape[0]):
        axs[experiment, 1].plot(
            X_star[experiment][..., 0],
            f_samples[i, ..., 1],
            color="orange",
            linewidth=0.2,
        )
    axs[experiment, 1].plot(X_star[experiment][..., 0], f_mean[..., 1], color="orange")


# show the plot
fig.tight_layout()
plt.show()

# %% [markdown]
"""
The results are visualised in a 2 $\times$ 3 plot with 6 subplots. The first row refers to experiments with few training data points and the second row to experiments with many training data points. The first column depicts the exact GPR model in green, the second column the SVGP model in purple and the third column the WSA model in orange. In each plot, training data points are marked in grey (as vertical bars in the first row and fill plots in the second row). We also assume the GPR model's prediction as ground truth, which is therefore plotted in all plots as black dashed lines (indicating mean +/- 2 * std).

In each plot, the model's prediction in terms of mean +/- 2 * std is plotted through fill plots, and function samples from the (approximate) posterior through thin solid lines (thick solid lines depict mean functions in the second and third column). Note that the coloured purple circles in the second column refer to predictions at the inducing point locations of the SVGP model.

It can be seen that, as training data points increase from the first to the second row, the predictions of the WSA model decrease drastically in areas relevant to extrapolation (i.e. close to but not inside the training data interval) because a lot of Fourier features would be required to accurately approximate a function sample drawn from a Matern kernel (because of its non-smooth nature). The same effect would be less severe for a function sample drawn from an RBF kernel that is smoother than a Matern kernel (and can hence be reliably approximated with fewer Fourier features). Note that the experiment is stochastic because the ground truth function sample from the prior kernel is random. There might be outcomes of the experiment in which the explained effect is less prominent than in other random outcomes -- so the last code block might require execution more than once to obtain a clear result.
"""
