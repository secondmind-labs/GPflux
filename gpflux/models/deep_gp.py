# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from typing import Optional, List, Iterator
from functools import reduce

import numpy as np
import tensorflow as tf
from scipy.stats import norm

import gpflow
from gpflow import settings
from gpflow.decors import params_as_tensors, autoflow
from gpflow.likelihoods import Gaussian
from gpflow.models.model import Model
from gpflow.params.dataholders import Minibatch, DataHolder

from gpflux.layers.latent_variable_layer import LatentVariableLayer, LatentVarMode


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

    def _get_Ws_iter(self, latent_var_mode: LatentVarMode, Ws=None) -> Iterator:
        """
        Returns slices from Ws for each of the Latent Variable Layers when
        LatentVarMode equals LatentVarMode.GIVEN, else returns None.
        """
        begin_index = 0
        for layer in self.layers:
            if latent_var_mode is not LatentVarMode.GIVEN:
                yield None  # we only return Ws when they are given
            elif not isinstance(layer, LatentVariableLayer):
                yield None  # we only return Ws for latent variable layers
            else:
                # select the right columns out of W
                end_index = begin_index + layer.latent_dim
                yield Ws[:, begin_index:end_index]
                # update indices
                begin_index = end_index

    @params_as_tensors
    def _build_decoder(self, X, full_cov=False, full_output_cov=False,
                       Ws=None, latent_var_mode=LatentVarMode.POSTERIOR):
        """
        :param X: N x W
        """
        X = tf.cast(X, dtype=settings.float_type)
        H = X

        Ws_iter = self._get_Ws_iter(latent_var_mode, Ws)

        for layer, W in zip(self.layers, Ws_iter):
            H, mean, cov = layer.propagate(
                H,
                W=W,
                X=X,
                full_cov=full_cov,
                full_output_cov=full_output_cov,
                latent_var_mode=latent_var_mode,
            )
        return mean, cov

    @params_as_tensors
    def _build_likelihood(self):
        f_mean, f_var = self._build_decoder(self.X)  # [N, P], [N, P]
        self.E_log_prob = tf.reduce_sum(
            self.likelihood.variational_expectations(f_mean, f_var, self.Y)
        )  # []

        # used for plotting in tensorboards:
        self.KL_U_sum = reduce(tf.add, (l.KL() for l in self.layers))  # []

        return tf.cast(
            self.E_log_prob * self.scale - self.KL_U_sum,
            settings.float_type
        )  # []

    def _predict_f(self, X):
        mean, variance = self._build_decoder(
            X,
            latent_var_mode=LatentVarMode.PRIOR
        )  # [N, P], [N, P]
        return mean, variance

    @params_as_tensors
    @autoflow([settings.float_type, [None, None]])
    def predict_y(self, X):
        mean, var = self._predict_f(X)
        return self.likelihood.predict_mean_and_var(mean, var)

    @autoflow([settings.float_type, [None, None]])
    def predict_f(self, X):
        return self._predict_f(X)

    @autoflow([settings.float_type, [None, None]],
              [settings.float_type, [None, None]])
    def predict_f_with_Ws(self, X, Ws):
        return self._build_decoder(
            X,
            Ws=Ws,
            latent_var_mode=LatentVarMode.GIVEN
        )

    @autoflow([settings.float_type, [None, None]],
              [settings.float_type, [None, None]])
    def predict_f_with_Ws_full_output_cov(self, X, Ws):
        return self._build_decoder(
            X,
            Ws=Ws,
            full_output_cov=True,
            latent_var_mode=LatentVarMode.GIVEN
        )

    @autoflow([settings.float_type, [None, None]],
              [settings.float_type, [None, None]])
    def predict_f_with_Ws_full_cov(self, X, Ws):
        return self._build_decoder(
            X,
            Ws=Ws,
            full_cov=True,
            latent_var_mode=LatentVarMode.GIVEN
        )

    @autoflow()
    def compute_KL_U_sum(self):
        return self.KL_U_sum

    @autoflow()
    def compute_data_fit(self):
        # check this works correctly with minibatching
        return self.E_log_prob * self.scale

    def log_pdf(self, X, Y):
        m, v = self.predict_y(X)
        ll = norm.logpdf(Y, loc=m, scale=v**0.5)
        # this assumes a Gaussian likelihood ?...
        return np.average(ll)

    def describe(self):
        """ High-level description of the model """
        desc = self.__class__.__name__
        desc += "\nLayers"
        desc += "\n------\n"
        desc += "\n".join(l.describe() for l in self.layers)
        desc += "\nlikelihood: " + self.likelihood.__class__.__name__
        return desc
