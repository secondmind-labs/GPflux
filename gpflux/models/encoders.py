import numpy as np
import tensorflow as tf

from typing import Callable, Optional, Union
from gpflow import settings, params_as_tensors, transforms

# from .layers import BaseLayer

from gpflow import Param, ParamList, Parameterized
from ..utils import xavier_weights

class Encoder(Parameterized):
    def __init__(self, D_in: int, latent_dim: int, name: Optional[str] = None):
        Parameterized.__init__(self, name=name)
        self.D_in = D_in
        self.latent_dim = latent_dim

    def __call__(self, Z: tf.Tensor) -> None:
        """ 
        Given Z, returns the mean and the log of the Cholesky
        of the latent variables (only the diagonal elements)
        In other words, w_n ~ N(m_n, exp(s_n)), where m_n, s_n = f(x_n).
        For this Encoder the function f is a NN.
        :return: N x latent_dim, N x latent_dim
        """
        raise NotImplementedError()


class GPflowEncoder(Encoder):
    def __init__(self,
                 D_in: int,
                 latent_dim: int,
                 network_dims: Union[np.ndarray, list],
                 activation_func = None,
                 name: Optional[str] = None):
        """
        Encoder that uses GPflow params to encode the features.
        :param network_dims: dimenions of inner MLPs
        """
        Encoder.__init__(self, D_in, latent_dim, name=name)

        self.network_dims = network_dims
        self.activation_func = activation_func or tf.nn.tanh
        self._build_network()

    def _build_network(self):
        Ws, bs = [], []
        dims = [self.D_in, *self.network_dims, self.latent_dim * 2]
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            Ws.append(Param(xavier_weights(dim_in, dim_out)))
            bs.append(Param(np.zeros(dim_out)))

        self.Ws, self.bs = ParamList(Ws), ParamList(bs)
    
    @params_as_tensors
    def __call__(self, Z: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            Z = tf.matmul(Z, W) + b
            if i < len(self.bs) - 1:
                Z = self.activation_func(Z)

        means, log_chol_diag = tf.split(Z, 2, axis=1)
        return means, log_chol_diag


class DirectlyParameterized(Parameterized):
    """
    Not compatible with minibatches
    """
    def __init__(self, num_data:int, latent_dim: int, mean: Optional[np.array]=None, name: Optional[str] = None):
        Parameterized.__init__(self,name=name)
        self.num_data = num_data
        self.latent_dim = latent_dim
        if mean is None:
            mean = np.random.randn(num_data, latent_dim)
        self.mean = Param(mean)
        self.std = Param(1e-5 * np.ones((num_data, latent_dim)), transform=transforms.positive)

    def __call__(self, Z: tf.Tensor) -> [tf.Tensor, tf.Tensor]:
        return self.mean, self.std


# class LatentLayer(BaseLayer):
#     def __init__(self, 
#                  latent_dimension: int, 
#                  encoder: Encoder) -> None:
#         """
#         Can only be the first layer of a Conditional Deep GP.
#         """
#         BaseLayer.__init__(self)
#         self.latent_dimension = latent_dimension
#         self.encoder = encoder

#     @params_as_tensors
#     def propagate(self, 
#                   X: tf.Tensor, 
#                   Y: tf.Tensor, 
#                   **kwargs) -> tf.Tensor:
#         """
#         :param X: N x Din
#         :param Y: N x Dout
#         :return network(X, Y), N x `latent_dimension`
#         """
#         return self.encoder(X, Y)

#     @params_as_tensors
#     def KL(self) -> tf.Tensor:
#         return tf.cast(0.0, settings.float_type)


#     def describe(self) -> str:
#         return "Latent layer with D = {}".format(self.latent_dimension)
