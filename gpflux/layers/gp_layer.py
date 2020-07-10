# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""A Sparse Variational Multioutput Gaussian Process Keras Layers"""

from typing import Optional

import numpy as np
import tensorflow as tf

from gpflow.kernels import MultioutputKernel
from gpflow.conditionals import sample_conditional
from gpflow.inducing_variables import MultioutputInducingVariables
from gpflow.kullback_leiblers import prior_kl
from gpflow.mean_functions import MeanFunction, Identity
from gpflow.utilities.bijectors import triangular
from gpflow import default_float, Parameter

from gpflux.layers import TrackableLayer
from gpflux.initializers import FeedForwardInitializer, Initializer
from gpflux.exceptions import GPInitializationError
from gpflux.utils.runtime_checks import verify_compatibility


class GPLayer(TrackableLayer):
    """A sparse variational multioutput GP layer"""

    def __init__(
        self,
        kernel: MultioutputKernel,
        inducing_variable: MultioutputInducingVariables,
        num_data: int,
        initializer: Optional[Initializer] = None,
        mean_function: Optional[MeanFunction] = None,
        *,
        returns_samples: bool = True,
        full_output_cov: bool = False,
        full_cov: bool = False,
        verify: bool = True,
        num_latent_gps: int = None,
        white: bool = True,
    ):
        """
        A sparse variational GP layer in whitened representation. This layer holds the
        kernel, variational parameters, inducing points and mean function.

        :param kernel: The multioutput kernel for the layer
        :param inducing_variable: The inducing features for the layer
        :param initializer: the initializer for the inducing variables and variational
            parameters. Default: FeedForwardInitializer
        :param mean_function: The mean function applied to the inputs. Default: Identity

        :param returns_samples: If True, return samples on calling the layer,
             Else return mean and variance
        :param full_cov: Use a full covariance in predictions, or just the diagonals
        :param full_output_cov: Return a full output covariance
        :param verify: if False, the call to `verify_compatibility` in the init is bypassed.
            The user is then responsible for making sure `kernel`, `mean_function`
            and `inducing_variable` are compatible and work togheter. It is also required
            to specify `num_latent_gps`, as this will not be infered from the other objects.
        :param num_latent_gps: number of (latent) GPs in the layer. Used to determine the size of
            the variational parameters `q_mu` and `q_sqrt`. Only required to be passed when
            `verify` is set to False, otherwise it is infered from the `kernel` and
            `inducing_variable`.
        :param white: determines the parameterisation of the inducing variables.
            If True: p(u) = N(0, I), else p(u) = N(0, Kuu).
            TODO(VD): The initializer currently only support white = True.
        """

        super().__init__(dtype=default_float())

        if initializer is None:
            initializer = FeedForwardInitializer()
        if mean_function is None:
            mean_function = Identity()

        self.kernel = kernel
        self.inducing_variable = inducing_variable
        self.initializer = initializer
        self.mean_function = mean_function

        self.returns_samples = returns_samples
        self.full_output_cov = full_output_cov
        self.full_cov = full_cov
        self.num_data = num_data
        self.white = white

        if verify:
            self.num_inducing, self.num_latent_gps = verify_compatibility(
                kernel, mean_function, inducing_variable
            )
        else:
            self.num_inducing, self.num_latent_gps = (
                len(inducing_variable),
                num_latent_gps,
            )

        self.q_mu = Parameter(
            np.zeros((self.num_inducing, self.num_latent_gps)),
            dtype=default_float(),
            name="q_mu",
        )  # [num_inducing, output_dim]

        self.q_sqrt = Parameter(
            np.stack([np.eye(self.num_inducing) for _ in range(self.num_latent_gps)]),
            transform=triangular(),
            dtype=default_float(),
            name="q_sqrt",
        )  # [output_dim, num_inducing, num_inducing]
        self._initialized = False

    def initialize_inducing_variables(self, **initializer_kwargs):
        if self._initialized:
            raise GPInitializationError("Initializing twice!")

        self.initializer.init_inducing_variable(
            self.inducing_variable, **initializer_kwargs
        )
        self._initialized = True

    def build(self, input_shape):
        """Build the variables necessary on first call"""

        super().build(input_shape)
        if not self.initializer.init_at_predict:
            self.initialize_inducing_variables()

    def predict(
        self,
        inputs,
        *,
        num_samples: Optional[int] = None,
        full_output_cov: bool = False,
        full_cov: bool = False,
        white: bool = True,
    ):
        """
        Make a prediction at N test inputs, with input_dim = D, output_dim = Q. Return a
        sample, and the conditional mean and covariance at these points.

        :param inputs: the inputs to predict at. shape [N, D]
        :param num_samples: the number of samples S, to draw.
            shape [S, N, Q] if S is not None else [N, Q].
        :param full_output_cov: If true: return the full covariance between Q ouput
            dimensions. Cov shape: -> [N, Q, N, Q]. If false: return block diagonal
            covariances. Cov shape: -> [Q, N, N]
        :param full_cov: If true: return the full (NxN) covariance for each output
            dimension Q.  Cov shape -> [Q, N, N]. If false: return variance (N) at each
            output dimension Q. Cov shape -> [N, Q]
        :param white:
        """
        if self.initializer.init_at_predict and not self._initialized:
            self.initialize_inducing_variables(inputs=inputs)

        mean_function = self.mean_function(inputs)
        sample_cond, mean_cond, cov = sample_conditional(
            inputs,
            self.inducing_variable,
            self.kernel,
            self.q_mu,
            q_sqrt=self.q_sqrt,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            white=white,
            num_samples=num_samples,
        )

        if num_samples is None:
            tf.debugging.assert_shapes(
                [(sample_cond, ["N", "Q"]), (mean_cond, ["N", "Q"])]
            )
        else:
            tf.debugging.assert_shapes(
                [(sample_cond, [num_samples, "N", "Q"]), (mean_cond, ["N", "Q"])]
            )

        samples = sample_cond + mean_function
        mean = mean_cond + mean_function

        return samples, mean, cov

    def call(self, inputs, training=False):
        """The default behaviour upon calling the GPLayer()(X)"""
        samples, mean, cov = self.predict(
            inputs,
            num_samples=None,
            full_output_cov=self.full_output_cov,
            full_cov=self.full_cov,
            white=self.white,
        )

        # TF quirk: add_loss must add a tensor to compile
        loss = (
            self.prior_kl(whiten=self.white)
            if training
            else tf.constant(0.0, dtype=default_float())
        )
        loss_per_datapoint = loss / self.num_data

        self.add_loss(loss_per_datapoint)
        self.add_metric(loss_per_datapoint, name="elbo_kl_gp", aggregation="mean")

        if self.returns_samples:
            return samples
        return mean, cov

    def prior_kl(self, whiten: bool = True):
        """
        The KL divergence from the variational distribution to the prior

        :param whiten:
        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I) independently for each GP
        """
        return prior_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=whiten
        )
