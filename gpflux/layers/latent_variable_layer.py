
import numpy as np
import tensorflow as tf

from gpflow import features, settings, params_as_tensors
from gpflow.conditionals import conditional, sample_conditional
from gpflow.kullback_leiblers import gauss_kl
from gpflow.mean_functions import Zero
from gpflow.params import Parameter, Parameterized

jitter_level = settings.numerics.jitter_level
float_type = settings.float_type

from .layers import BaseLayer
from ..models.encoders import GPflowEncoder

from enum import Enum

class LatentVarMode(Enum):
    """
    We need to distinguish between training and test points when propagating with latent variables. We have
    a parameterized variational posterior for the N data, but at test points we might want to do one of three things:
    """
    # sample from N(0, 1)
    PRIOR = 1

    # we are dealing with the N observed data points, so we use the vatiational posterior
    POSTERIOR = 2

    # for plotting purposes, it is useful to have a mechanism for setting W to fixed values, e.g. on a grid
    GIVEN = 3


class LatentVariableLayer(BaseLayer):
    """
    A latent variable layer, with amortized mean-field VI 
    
    The prior is N(0, 1), and inference is factorised N(a, b), where a, b come from 
    an encoder network.
    
    When propagating there are two possibilities:
    1) We're doing inference, so we use the variational distribution
    2) We're looking at test points, so we use the prior
    
    """
    def __init__(self, latent_variables_dim, XY_dim=None, encoder=None):
        BaseLayer.__init__(self)
        self.latent_variables_dim = latent_variables_dim

        if not encoder:
            assert XY_dim, 'must pass XY_dim or else an encoder'
            encoder = GPflowEncoder(XY_dim,
                                    latent_variables_dim,
                                    [10, 10])
        self.encoder = encoder
        self.is_encoded = False

    def encode_once(self):
        if not self.is_encoded:
            XY = tf.concat([self.root.X, self.root.Y], 1)
            q_mu, log_q_sqrt = self.encoder(XY)
            self.q_mu = q_mu
            self.q_sqrt = tf.nn.softplus(log_q_sqrt - 3.)  # bias it towards small vals at first
            self.is_encoded = True

    def KL(self):
        self.encode_once()
        return gauss_kl(self.q_mu, self.q_sqrt)

    def describe(self):
        """ describes the key properties of a layer """
        return "LatentVarLayer: with dim={}".format(self.latent_variables_dim)

    def propagate(self, X, sampling=True, W=None, **kwargs):
        raise NotImplementedError



class LatentVariableConcatLayer(LatentVariableLayer):
    """
    A latent variable layer where the latents are concatenated with the input
    """
    @params_as_tensors
    def propagate(self, X, sampling=True, latent_var_mode=LatentVarMode.POSTERIOR, W=None,
                  full_cov=False, full_output_cov=False):
        self.encode_once()
        if sampling:
            if latent_var_mode == LatentVarMode.POSTERIOR:
                z= tf.random_normal(tf.shape(self.q_mu), dtype=float_type)
                W = self.q_mu + z * self.q_sqrt

            elif latent_var_mode == LatentVarMode.PRIOR:
                W = tf.random_normal([tf.shape(X)[0], self.latent_variables_dim], dtype=float_type)

            elif latent_var_mode == LatentVarMode.GIVEN:
                assert isinstance(W, tf.Tensor)


            return tf.concat([X, W], 1)

        else:

            if latent_var_mode == LatentVarMode.POSTERIOR:
                XW_mean = tf.concat([X, self.q_mu], 1)
                XW_var = tf.concat([tf.zeros_like(X), self.q_sqrt ** 2])
                return XW_mean, XW_var

            elif latent_var_mode == LatentVarMode.PRIOR:
                z = tf.zeros([tf.shape(X)[0], self.latent_variables_dim], dtype=float_type)
                o = tf.ones([tf.shape(X)[0], self.latent_variables_dim], dtype=float_type)
                XW_mean = tf.concat([X, z], 1)
                XW_var = tf.concat([tf.zeros_like(X), o])
                return XW_mean, XW_var

            else:
                raise NotImplementedError


# class LatentVariableAdditiveLayer(LatentVariableLayer):
#     """
#     A latent variable layer where the latents are added to the input
#     """
#     @params_as_tensors
#     def propagate(self, X, sampling=True, W=None, **kwargs):
#         self.encode_once()
#         if sampling:
#             if W is None:
#                 z = tf.random_normal(tf.shape(self.q_mu), dtype=float_type)
#                 W = self.q_mu + z * self.q_sqrt
#             return X + W
#
#         else:
#             return X + self.q_mu, self.q_sqrt**2