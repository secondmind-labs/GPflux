# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


from typing import Optional, List, Iterator
from functools import reduce

import numpy as np
import tensorflow as tf

import gpflow
from gpflow import settings
from gpflow.decors import params_as_tensors, autoflow
from gpflow.likelihoods import Gaussian
from gpflow.models.model import Model
from gpflow.params.dataholders import Minibatch, DataHolder

from gpflux.layers import AbstractLayer
from gpflux.layers.base_layer import LayerOutput
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
                 layers: List[AbstractLayer], *,
                 likelihood: Optional[gpflow.likelihoods.Likelihood] = None,
                 minibatch_size: Optional[int] = None,
                 name: Optional[str] = None):
        """
        :param X: np.ndarray, N x Dx
        :param Y: np.ndarray, N x Dy
        :param layers: list
            List of `layers.AbstractLayer` instances,
            e.g. PerceptronLayer, ConvLayer, GPLayer, ...
        :param likelihood: gpflow.likelihoods.Likelihood object
            Analytic expressions exists for the Gaussian case.
        :param minibatch_size: int
        """
        Model.__init__(self, name=name)

        assert X.ndim == 2
        assert Y.ndim == 2

        self.num_data = X.shape[0]
        self.layers = gpflow.ParamList(layers)
        self.likelihood = Gaussian() if likelihood is None else likelihood

        if minibatch_size and (minibatch_size > 0) and (minibatch_size < X.shape[0]):
            self.X = Minibatch(X, batch_size=minibatch_size, seed=0)
            self.Y = Minibatch(Y, batch_size=minibatch_size, seed=0)
            self.scale = self.num_data / minibatch_size
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
    def _build_decoder(
            self,
            X,
            *,
            Y=None,
            full_cov=False,
            full_output_cov=False,
            Ws=None,
            latent_var_mode=LatentVarMode.POSTERIOR
    ) -> List[LayerOutput]:
        """
        :param X: N x W
        """
        X = tf.cast(X, dtype=settings.float_type)
        if Y is not None:
            Y = tf.cast(Y, dtype=settings.float_type)

        H = X
        layer_outputs = []
        Ws_iter = self._get_Ws_iter(latent_var_mode, Ws)

        for layer, W in zip(self.layers, Ws_iter):
            layer_output = layer.propagate(
                H,
                W=W,
                X=X,
                Y=Y,
                full_cov=full_cov,
                full_output_cov=full_output_cov,
                latent_var_mode=latent_var_mode,
            )
            H = layer_output.sample
            layer_outputs.append(layer_output)

        return layer_outputs

    @params_as_tensors
    def _build_likelihood(self):

        layer_outputs = self._build_decoder(self.X, Y=self.Y)
        f_mean, f_var = layer_outputs[-1].mean, layer_outputs[-1].covariance  # [N, P], [N, P]

        var_exp = self.likelihood.variational_expectations(f_mean, f_var, self.Y)  # [N, 1]
        var_exp_sum = tf.reduce_sum(var_exp)

        global_kls = reduce(tf.add, [o.global_kl for o in layer_outputs])  # []
        local_kls = reduce(tf.add, [o.local_kl for o in layer_outputs])  # [N]
        local_kls_sum = tf.reduce_sum(local_kls)

        elbo = (var_exp_sum - local_kls_sum) * self.scale - global_kls

        return tf.cast(elbo, settings.float_type)

    def _predict_f(self, X):
        layer_outputs = self._build_decoder(
            X,
            latent_var_mode=LatentVarMode.PRIOR
        )  # [N, P], [N, P]
        return layer_outputs[-1].mean, layer_outputs[-1].covariance

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
        layer_outputs = self._build_decoder(
            X,
            Ws=Ws,
            latent_var_mode=LatentVarMode.GIVEN
        )
        return layer_outputs[-1].mean, layer_outputs[-1].covariance

    @autoflow([settings.float_type, [None, None]],
              [settings.float_type, [None, None]])
    def predict_f_with_Ws_full_output_cov(self, X, Ws):
        layer_outputs = self._build_decoder(
            X,
            Ws=Ws,
            full_output_cov=True,
            latent_var_mode=LatentVarMode.GIVEN
        )
        return layer_outputs[-1].mean, layer_outputs[-1].covariance

    @autoflow([settings.float_type, [None, None]],
              [settings.float_type, [None, None]])
    def predict_f_with_Ws_full_cov(self, X, Ws):
        layer_outputs = self._build_decoder(
            X,
            Ws=Ws,
            full_cov=True,
            latent_var_mode=LatentVarMode.GIVEN
        )
        return layer_outputs[-1].mean, layer_outputs[-1].covariance

    def describe(self):
        """ High-level description of the model """
        desc = self.__class__.__name__
        desc += "\nLayers"
        desc += "\n------\n"
        desc += "\n".join(l.describe() for l in self.layers)
        desc += "\nlikelihood: " + self.likelihood.__class__.__name__
        return desc
