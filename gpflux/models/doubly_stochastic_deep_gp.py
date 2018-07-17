import gpflow
import numpy as np
import tensorflow as tf

from scipy.stats import norm
from functools import reduce

from gpflow.decors import params_as_tensors, autoflow
from gpflow.likelihoods import Gaussian
from gpflow.models.model import Model
from gpflow.params.dataholders import Minibatch

float_type = gpflow.settings.float_type
int_type = gpflow.settings.int_type
jitter_level = gpflow.settings.numerics.jitter_level


class DeepGP(Model):
    """
    @inproceedings{salimbeni2017doubly,
        title={Doubly Stochastic Variational Inference for Deep Gaussian Processes},
        author={Salimbeni, Hugh and Deisenroth, Marc},
        booktitle={NIPS},
        year={2017}
    }
    """

    def __init__(self, X, Y, layers, likelihood=None, batch_size=None, name=None):
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
        self.likelihood = likelihood or Gaussian()

        if (batch_size is not None) and (batch_size > 0):
            self.X = Minibatch(X, batch_size=batch_size, seed=0)
            self.Y = Minibatch(Y, batch_size=batch_size, seed=0)
            self.scale = self.num_data / batch_size
        else:
            self.X = X
            self.Y = Y
            self.scale = 1.0

    def _build_decoder(self, Z):
        """
        :param Z: N x W
        """
        Z = tf.cast(Z, dtype=tf.float64)
        for layer in self.layers[:-1]:
            Z = layer.propagate(Z, sampling=True, full_output_cov=False, full_cov=False)

        f_mean, f_var = self.layers[-1].propagate(Z, sampling=False, full_output_cov=False, full_cov=False)
        return f_mean, f_var

    @params_as_tensors
    def _build_likelihood(self):

        f_mean, f_var = self._build_decoder(self.X)  # N x P, N x P
        self.E_log_prob = tf.reduce_sum(self.likelihood.variational_expectations(f_mean, f_var, self.Y))

        self.KL_U_layers = reduce(tf.add, (l.KL() for l in self.layers))

        ELBO = self.E_log_prob * self.scale - self.KL_U_layers
        return tf.cast(ELBO, float_type)

    def _predict_f(self, X):
        mean, variance = self._build_decoder(X)  # N x P, N x P
        return mean, variance

    @params_as_tensors
    @autoflow([float_type, [None, None]])
    def predict_y(self, X):
        mean, var = self._predict_f(X)
        return self.likelihood.predict_mean_and_var(mean, var)

    @autoflow([float_type, [None, None]])
    def predict_f(self, X):
        return self._predict_f(X)

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


    # @autoflow([float_type, [None, None]])
    # def encode(self, X):
    #     return self._eval_encoder(X)  # N x W, N x W

    # @autoflow([float_type, [None, None]])
    # def decode(self, Z):
    #     mean, _ = self._build_decoder(Z, full_cov_output=False)  # N x P, N x P
    #     return mean

    # @autoflow([float_type, [None, None]])
    # def decode_full_cov_output(self, Z):
    #     mean, var = self._build_decoder(Z, full_cov_output=True)  # N x P, N x P x P
    #     N, P = tf.shape(mean)[0], tf.shape(mean)[1]
    #     jittermat = jitter_level * tf.eye(P, batch_shape=[N], dtype=float_type)  # N x P x P
    #     eps = tf.random_normal((N, P, 1), dtype=float_type)  # N x P x 1
    #     return mean, var, mean + tf.matmul(tf.cholesky(var + jittermat), eps)[..., 0]  # N x P, N x P

    # @autoflow([float_type, [None, None]])
    # def decode_dense(self, Z):
    #     return self.dense_layer.sample_conditional(Z)  # N x P

    # @params_as_tensors
    # @autoflow((tf.float32, [None, None]),
    #           (int_type, ))
    # def compute_test_log_likelihood(self, Xs, num):
    #     Xs = tf.cast(Xs, dtype=tf.float64)
    #     NUM = num  # M
    #     N = tf.shape(Xs)[0]
    #     Zs = tf.random_normal((N * NUM, self.latent_dim), dtype=tf.float64)  # N M x L
    #     Xs_recon_mean, Xs_recon_var = self._build_decoder(Zs, full_cov_output=False)  # N M x Dx, N M x Dx

    #     Xs_repeat = tf.tile(Xs[:, None, :], [1, NUM, 1])  # N x M x Dx
    #     Xs = tf.reshape(Xs_repeat, [N * NUM, self.X_dim])  # N M x Dx

    #     p = self.likelihood.predict_density(Xs_recon_mean, Xs_recon_var, Xs)  # N M x Dx
    #     p = tf.reshape(tf.reduce_sum(p, axis=1), [N, NUM])  # N x M
    #     p = tf.reduce_logsumexp(p, axis=1) - tf.log(tf.cast(NUM, dtype=tf.float64))  # N
    #     return p

    # # def batch_predict_density_sampled(self, Xs, num_samples=10, batchsize=100):
    # #     ls = []
    # #     Ns = len(Xs)
    # #     splits = int(Ns/batchsize)
    # #     for xs in np.array_split(Xs, splits):
    # #         l = self.compute_test_log_likelihood(xs, num_samples)
    # #         ls.append(l)
    # #     return np.average(np.concatenate(ls, 0))

    # def batch_compute_test_ll(self, Xs, num_samples=10, size=200):
    #     n, tll = len(Xs) // size, []
    #     if n == 0 and len(Xs) > 0: n = 1

    #     for i in range(n):
    #         b, e = size * i, min(size * (i+1), len(Xs))
    #         tll.append(self.compute_test_log_likelihood(Xs[b:e], num_samples).flatten())

    #     return np.average(np.concatenate(tll, axis=0))
