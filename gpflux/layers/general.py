"""
Replicates certain functionality from bayesfunc library.
"""
from typing import Dict, Any, List, Tuple

import tensorflow as tf
from gpflow import Parameter, default_float
from gpflow.base import Module, TensorType
from gpflow.utilities import set_trainable
from gpflow.utilities.bijectors import positive


class KG:
    def __init__(self, ii: TensorType, it: TensorType, tt: TensorType):
        self.ii = ii
        self.it = it
        self.tt = tt


class Kernel(tf.keras.layers.Layer):
    def __init__(self, trainable_noise=False):
        super().__init__()
        if trainable_noise:
            self.noise = Parameter(1e-5*tf.ones([]), transform=positive(), dtype=default_float())
        else:
            self.noise = None

    def call(
        self,
        xG: KG,
        *args: List[Any],
        **kwargs: Dict[str, Any]
    ) -> KG:
        (d2ii, d2it, d2tt) = self.distances(xG, full_cov=kwargs.get("full_cov"))

        if self.noise is not None:
            noise = self.noise
        else:
            noise = 0

        Kii = self.kernel(d2ii)
        Kii = Kii + noise * tf.eye(tf.shape(Kii)[-1], dtype=default_float())
        Kit = self.kernel(d2it)
        if kwargs.get("full_cov"):
            Ktt = self.kernel(d2tt)
            Ktt = Ktt + noise * tf.eye(tf.shape(Ktt)[-1], dtype=default_float())
        else:
            Ktt = self.kernel(d2tt) + noise

        h = self.height

        return KG(h*Kii, h*Kit, h*Ktt)


class KernelGram(Kernel):
    def __init__(self, lengthscale=1., height=1., trainable_noise=False, train_lengthscale=True):
        super().__init__(trainable_noise=trainable_noise)
        self.lengthscales = Parameter(lengthscale*tf.ones([]), transform=positive(),
                                      dtype=default_float())
        if not train_lengthscale:
            set_trainable(self.lengthscales, False)

        self.height = Parameter(height*tf.ones([]), transform=positive(), dtype=default_float())

    def distances(self, G: KG, full_cov: bool =False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        Gii = G.ii
        Git = G.it
        Gtt = G.tt

        diag_Gii = tf.linalg.diag_part(Gii)
        d2ii = diag_Gii[..., :, None] + diag_Gii[..., None, :] - 2*Gii
        if full_cov:
            diag_Gtt = tf.linalg.diag_part(Gtt)
            d2it = diag_Gii[..., :, None] + diag_Gtt[..., None, :] - 2 * Git
            d2tt = diag_Gtt[..., :, None] + diag_Gtt[..., None, :] - 2 * Gtt
        else:
            d2it = diag_Gii[..., :, None] + Gtt[..., None, :] - 2*Git
            d2tt = tf.zeros_like(Gtt)

        lm2 = 1/(self.lengthscales**2)

        return lm2*d2ii, lm2*d2it, lm2*d2tt


class SqExpKernelGram(KernelGram):
    def __init__(self, height=1., trainable_noise=False, train_lengthscale=True):
        super().__init__(height=height, trainable_noise=trainable_noise, train_lengthscale=train_lengthscale)

    def kernel(self, d2):
        return tf.exp(-0.5*d2)


class FeaturesToKernelARD(tf.keras.layers.Layer):
    def __init__(self, num_inducing, in_features, epsilon=None):
        super().__init__()
        self.num_inducing = num_inducing
        self.in_features = in_features
        self.epsilon = epsilon
        self.lengthscales = Parameter(tf.ones([in_features]), transform=positive(), dtype=default_float())

    def call(
        self,
        x: TensorType,
        *args: List[Any],
        **kwargs: Dict[str, Any]
    ) -> KG:
        x = x/self.lengthscales
        xi = x[:, :self.num_inducing]
        xt = x[:, self.num_inducing:]

        ii = tf.linalg.matmul(xi, xi, transpose_b=True) / self.in_features
        it = tf.linalg.matmul(xi, xt, transpose_b=True) / self.in_features
        if kwargs.get("full_cov"):
            tt = tf.linalg.matmul(xt, xt, transpose_b=True) / self.in_features
        else:
            tt = tf.reduce_sum(xt ** 2, -1) / self.in_features

        if self.epsilon is not None:
            ii = ii + self.epsilon * tf.eye(tf.shape(ii)[-1], dtype=default_float())
            if kwargs.get("full_cov"):
                tt = tt + self.epsilon * tf.eye(tf.shape(tt)[-1], dtype=default_float())
            else:
                tt = tt + self.epsilon

        return KG(ii, it, tt)
