# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import gpflow
import numpy as np
import tensorflow as tf

from scipy.stats import norm
from functools import reduce

from typing import Optional, List

from gpflow import settings, Param
from gpflow.decors import params_as_tensors, autoflow
from gpflow.likelihoods import Gaussian
from gpflow.models.model import Model
from gpflow.params.dataholders import Minibatch, DataHolder

from ..layers.latent_variable_layer import LatentVariableLayer, LatentVarMode


class DeepGP(Model):
    """
    Implementation of a Deep Gaussian process, following the specification of:

    @inproceedings{salimbeni2017doubly,
        title={Doubly Stochastic Variational Inference for Deep Gaussian Processes},
        author={Salimbeni, Hugh and Deisenroth, Marc},
        booktitle={NIPS},
        year={2017}
    }
    """
    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 layers: List, *,
                 likelihood: Optional[gpflow.likelihoods.Likelihood] = None,
                 batch_size: Optional[int] = None,
                 name: Optional[str] = None):
        """
        :param X: np.ndarray, N x Dx
        :param Y: np.ndarray, N x Dy
        :param layers: list
            List of `layers.BaseLayer` instances, e.g. PerceptronLayer, ConvLayer, GPLayer, ...
        :param likelihood: gpflow.likelihoods.Likelihood object
            Analytic expressions exists for the Gaussian case.
        :param batch_size: int
        """
        Model.__init__(self, name=name)

        assert X.ndim == 2
        assert Y.ndim == 2

        self.num_data = X.shape[0]
        self.layers = gpflow.ParamList(layers)
        self.likelihood = Gaussian() if likelihood is None else likelihood

        if (batch_size is not None) and (batch_size > 0) and (batch_size < X.shape[0]):
            self.X = Minibatch(X, batch_size=batch_size, seed=0)
            self.Y = Minibatch(Y, batch_size=batch_size, seed=0)
            self.scale = self.num_data / batch_size
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)
            self.scale = 1.0
        
        self.alpha = Param(1.0)
        self.alpha.set_trainable(False)

    def _get_Ws_iter(self, latent_var_mode: LatentVarMode, Ws=None) -> iter:
        i = 0
        for layer in self.layers:
            if latent_var_mode == LatentVarMode.GIVEN and isinstance(layer, LatentVariableLayer):

                # passing some fixed Ws, which are packed to a single tensor for ease of use with autoflow
                assert isinstance(Ws, tf.Tensor)
                d = layer.latent_dim
                yield Ws[:, i:(i+d)]
                i += d
            else:
                yield None

    @params_as_tensors
    def _build_decoder(self, Z, full_cov=False, full_output_cov=False,
                       Ws=None, latent_var_mode=LatentVarMode.POSTERIOR):
        """
        :param Z: N x W
        """
        Z = tf.cast(Z, dtype=settings.float_type)

        Ws_iter = self._get_Ws_iter(latent_var_mode, Ws)  # iter, returning either None or slices from Ws

        for layer, W in zip(self.layers[:-1], Ws_iter):
            Z = layer.propagate(Z,
                                sampling=True,
                                W=W,
                                latent_var_mode=latent_var_mode,
                                full_output_cov=full_output_cov,
                                full_cov=full_cov)

        return self.layers[-1].propagate(Z,
                                         sampling=False,
                                         W=next(Ws_iter),
                                         latent_var_mode=latent_var_mode,
                                         full_output_cov=full_output_cov,
                                         full_cov=full_cov)  #f_mean, f_var

    @params_as_tensors
    def _build_likelihood(self):
        f_mean, f_var = self._build_decoder(self.X)  # N x P, N x P
        self.E_log_prob = tf.reduce_sum(self.likelihood.variational_expectations(f_mean, f_var, self.Y))

        # self.KL_U_layers = reduce(tf.add, (l.KL() for l in self.layers))
        self.KLs_global = reduce(tf.add, (l.KL_global() for l in self.layers))
        self.KLs_minibatch = reduce(tf.add, (l.KL_minibatch() for l in self.layers))

        self.KLs_global = tf.check_numerics(self.KLs_global, 'KL global NAN')
        self.KLs_minibatch = tf.check_numerics(self.KLs_minibatch, 'KL minibatch NAN')

        ELBO = (self.E_log_prob - self.KLs_minibatch) * self.scale - self.KLs_global

        ELBO = tf.check_numerics(ELBO, 'nan in ELBO')

        return tf.cast(ELBO, settings.float_type)

    def _predict_f(self, X):
        mean, variance = self._build_decoder(X, latent_var_mode=LatentVarMode.PRIOR)  # N x P, N x P
        return mean, variance

    @params_as_tensors
    @autoflow([settings.float_type, [None, None]])
    def predict_y(self, X):
        mean, var = self._predict_f(X)
        return self.likelihood.predict_mean_and_var(mean, var)

    @autoflow([settings.float_type, [None, None]])
    def predict_f(self, X):
        return self._predict_f(X)

    @autoflow([settings.float_type, [None, None]], [settings.float_type, [None, None]])
    def predict_f_with_Ws(self, X, Ws):
        return self._build_decoder(X, Ws=Ws, latent_var_mode=LatentVarMode.GIVEN)

    @autoflow([settings.float_type, [None, None]], [settings.float_type, [None, None]])
    def predict_f_with_Ws_full_output_cov(self, X, Ws):
        return self._build_decoder(X, Ws=Ws, full_output_cov=True, latent_var_mode=LatentVarMode.GIVEN)

    @autoflow([settings.float_type, [None, None]], [settings.float_type, [None, None]])
    def predict_ys_with_Ws_full_output_cov(self, X, Ws):
        Fm, Fv = self._build_decoder(X, Ws=Ws, full_output_cov=True, latent_var_mode=LatentVarMode.GIVEN)
        Fs = gpflow.conditionals._sample_mvn(Fm, Fv, "full")
        return self.likelihood.conditional_mean(Fs)

    @autoflow([settings.float_type, [None, None]], [settings.float_type, [None, None]])
    def predict_f_with_Ws_full_cov(self, X, Ws):
        return self._build_decoder(X, Ws=Ws, full_cov=True, latent_var_mode=LatentVarMode.GIVEN)

    @autoflow()
    def compute_KL_global(self):
        return self.KLs_global

    @autoflow()
    def compute_KL_minibatch(self):
        return self.scale * self.KLs_minibatch

    @autoflow([settings.float_type, [None, None]], [settings.float_type, [None, None]])
    def decode_inner_layer(self, X, W):
        Z = self.layers[0].propagate(X, sampling=True, W=W, latent_var_mode=LatentVarMode.GIVEN)
        Z = self.layers[1].propagate(Z, sampling=True, W=None, full_cov=False, full_output_cov=True)
        return Z

    @autoflow([settings.float_type, [None, None]],
              [settings.int_type, ] )
    def nll(self, Y, num):
        """
        X: empty shape of Ws
        Ws: Nw x L, Nw: monte carlo samples, L: latent dim
        Y: Ns x D, NS: num test points, D output dim
        """
        # m is Nw x D, and v is Nw x D x D

        # debug_op = tf.py_func(_debug_func2, [Ws], [tf.bool])
        # with tf.control_dependencies(debug_op):
        #     Ws = tf.identity(Ws, name='out')

        X = tf.zeros([num, 0], dtype=settings.float_type)
        m, v = self._build_decoder(X,
                                   full_cov=False,
                                   full_output_cov=True,
                                   latent_var_mode=LatentVarMode.PRIOR)
        # P is Nw x D
        P = self.likelihood.predict_mean_from_f_full_output_cov(m, v)
        Pr = tf.tile(P[None, ...], multiples=[tf.shape(Y)[0], 1, 1])  # Ns x Nw x D

        # val = tf.reduce_sum(tf.is_nan(Pr))
        # Pr = tf.Print(Pr, ["nans in Pr", val])

        def _debug_func(p, l, y):
            import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
            from IPython import embed; embed()  # XXX DEBUG
            return False


        Yr = tf.tile(Y[:, None, :], multiples=[1, num, 1])  # Ns x Nw x D
        L = self.likelihood.eval(Pr, Yr)

        # debug_op = tf.py_func(_debug_func, [Pr, L, Y], [tf.bool])
        # with tf.control_dependencies(debug_op):
        #     L = tf.identity(L, name='out')

        # val = tf.reduce_sum(tf.is_nan(Pr))
        # Pr = tf.Print(Pr, ["nans in Pr", val])

        L = tf.reduce_sum(L, axis=2)  # Ns x Nw
        L = tf.reduce_logsumexp(L, axis=1) - tf.log(tf.cast(num, tf.float64))  # Ns
        return - tf.reduce_mean(L)


    @autoflow()
    def compute_KL_U(self):
        return self.KL_U_layers

    @autoflow()
    def compute_data_fit(self):
        return self.E_log_prob * self.scale

    def log_pdf(self, X, Y):
        m, v = self.predict_y(X)
        l = norm.logpdf(Y, loc=m, scale=v**0.5)
        return np.average(l)

    def describe(self):
        """ High-level description of the model """
        desc = self.__class__.__name__
        desc += "\nLayers"
        desc += "\n------\n"
        desc += "\n".join(l.describe() for l in self.layers)
        desc += "\nlikelihood: " + self.likelihood.__class__.__name__
        return desc


