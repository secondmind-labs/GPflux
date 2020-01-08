# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""A Sparse Variational Multioutput Gaussian Process Keras Layers"""

from typing import Optional

import numpy as np

from gpflow.kernels import (
    MultioutputKernel,
    SharedIndependent,
    SeparateIndependent,
    LinearCoregionalization,
)
from gpflow.conditionals import sample_conditional
from gpflow.inducing_variables import (
    MultioutputInducingVariables,
    FallbackSeparateIndependentInducingVariables,
)
from gpflow.kullback_leiblers import prior_kl
from gpflow.mean_functions import MeanFunction, Identity
from gpflow.utilities.bijectors import triangular
from gpflow import default_float, Parameter

from gpflux2.layers import TrackableLayer
from gpflux2.initializers import FeedForwardInitializer, Initializer


_DEFAULT_INITIALIZER = 'FeedForwardInitializer'
_DEFAULT_MEAN_FUNCTION = 'Identity'

class GPLayer(TrackableLayer):
    """A sparse variational multioutput GP layer"""

    def __init__(
        self,
        kernel: MultioutputKernel,
        inducing_variable: MultioutputInducingVariables,
        initializer: Optional[Initializer] = _DEFAULT_INITIALIZER,
        mean_function: Optional[MeanFunction] = _DEFAULT_MEAN_FUNCTION,
        *,
        use_samples: bool = True,
        full_output_cov: bool = False,
        full_cov: bool = False,
    ):
        """
        A sparse variational GP layer in whitened representation. This layer holds the
        kernel, variational parameters, inducing points and mean function.

        :param kernel: The multioutput kernel for the layer
        :param inducing_variable: The inducing features for the layer
        :param initializer: the initializer for the inducing variables and variational
            parameters. Default: FeedForwardInitializer
        :param mean_function: The mean function applied to the inputs. Default: Identity

        :param use_samples: If True, return samples on calling the layer,
             Else return mean and variance
        :param full_cov: Use a full covariance in predictions, or just the diagonals
        :param full_output_cov: Return a full output covariance
        """

        super().__init__(dtype=default_float())

        if initializer is _DEFAULT_INITIALIZER:
            initializer = FeedForwardInitializer()
        if mean_function is _DEFAULT_MEAN_FUNCTION:
            mean_function = Identity()

        self.kernel = kernel
        self.inducing_variable = inducing_variable
        self.initializer = initializer
        self.mean_function = mean_function

        self.use_samples = use_samples
        self.full_output_cov = full_output_cov
        self.full_cov = full_cov

        self.num_inducing, self.output_dims = self.verify_dims(
            kernel, mean_function, inducing_variable
        )

        self.q_mu = Parameter(
            np.zeros((self.num_inducing, self.output_dims)),
            dtype=default_float(),
            name="q_mu",
        )  # [num_inducing, output_dim]

        self.q_sqrt = Parameter(
            np.zeros((self.output_dims, self.num_inducing, self.num_inducing)),
            transform=triangular(),
            dtype=default_float(),
            name="q_sqrt",
        )  # [output_dim, num_inducing, num_inducing]
        self._initialized = False

    @staticmethod
    def verify_dims(
        kernel: MultioutputKernel,
        mean_function: MeanFunction,
        inducing_variable: MultioutputInducingVariables,
    ):
        """
        Provide error checking on shapes at layer construction. This method will be
        made simpler by having enhancements to GPflow: eg by adding foo.output_dim
        attribute, where foo is a MultioutputInducingPoints

        :param kernel: The multioutput kernel for the layer
        :param inducing_variable: The inducing features for the layer
        :param mean_function: The mean function applied to the inputs.
        """
        if not isinstance(inducing_variable, MultioutputInducingVariables):
            raise TypeError(
                "`inducing_variable` must be a `MultioutputInducingVariables`"
            )
        if not isinstance(kernel, MultioutputKernel):
            raise TypeError("`kernel` must be a `MultioutputKernel`")

        if isinstance(inducing_variable, FallbackSeparateIndependentInducingVariables):
            inducing_output_dim = len(inducing_variable.inducing_variable_list)
        else:
            inducing_output_dim = None

        if isinstance(kernel, SharedIndependent):
            kernel_output_dim = kernel.P
        if isinstance(kernel, (SeparateIndependent, LinearCoregionalization)):
            kernel_output_dim = len(kernel.kernels)

        if inducing_output_dim is not None:
            assert kernel_output_dim == inducing_output_dim

        num_inducing_points = len(inducing_variable) # currently the same for each dim
        return num_inducing_points, kernel_output_dim

    def _init_at_build(self, input_shape):
        self.initializer.init_variational_params(self.q_mu, self.q_sqrt)

        if not self.initializer.init_at_predict:
            self.initializer.init_inducing_variable(self.inducing_variable)
            self._initialized = True

    def _init_at_predict(self, inputs):
        if self.initializer.init_at_predict and not self._initialized:
            self.initializer.init_inducing_variable(self.inducing_variable, inputs)
            self._initialized = True

    def build(self, input_shape):
        """Build the variables necessary on first call"""
        super().build(input_shape)

        if self.initializer is not None:
            self._init_at_build(input_shape)

        self.add_loss(self.prior_kl)

    def predict(
        self,
        inputs,
        *,
        num_samples: Optional[int]=None,
        full_output_cov: bool=False,
        full_cov: bool=False,
        white: bool=True,
    ):
        """
        Make a prediction at N test inputs, with input_dim = D, output_dim=Q. Return a
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
        if self.initializer is not None:
            self._init_at_predict(inputs)

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

        samples = sample_cond + mean_function  # [S, N, Q] if S not None else [N, Q]
        mean = mean_cond + mean_function  # [N, Q]

        return samples, mean, cov

    def call(self, inputs, *args, **kwargs):
        """The default behaviour upon calling the GPLayer()(X)"""
        samples, mean, cov = self.predict(
            inputs,
            num_samples=None,
            full_output_cov=self.full_output_cov,
            full_cov=self.full_cov,
        )
        if self.use_samples:
            return samples
        return mean, cov

    def prior_kl(self, whiten:bool=True):
        """
        The KL divergence from the variational distribution to the prior

        :param whiten:
        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I) independently for each GP
        """
        return prior_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=whiten
        )
