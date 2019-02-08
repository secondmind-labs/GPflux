from abc import ABC

import gpflow
import tensorflow as tf


# Note on the current implementation
# There are two things which are unsatisfactory:
#  - We need `input_dim`, when often this is determined by the `basekern`.
#  - We use multiple inheritance to define the orbits
# Instead, we should make an `Orbit` class, which is passed to the constructor.


class InvariantBase(gpflow.kernels.Kernel, ABC):
    def __init__(self, input_dim, basekern, orbit, **kwargs):
        super().__init__(input_dim, **kwargs)
        self.basekern = basekern
        self.orbit = orbit


class Invariant(InvariantBase):
    """
    Invariant kernel that exactly computes the kernel matrix.
    We're keeping this mainly for testing.
    """

    @gpflow.decors.params_as_tensors
    def K(self, X, X2=None):
        Xp = self.orbit.get_full_orbit(X)
        N, num_orbit_points = tf.shape(Xp)[0], tf.shape(Xp)[1]
        Xp = tf.reshape(Xp, (-1, self.basekern.input_dim))
        Xp2 = None
        if X2 is not None:
            Xp2 = tf.reshape(self.orbit.get_full_orbit(X2), (-1, self.basekern.input_dim))

        bigK = self.basekern.K(Xp, Xp2)  # N * num_patches x N * num_patches
        K = tf.reduce_mean(tf.reshape(bigK, (N, num_orbit_points, -1, num_orbit_points)), [1, 3])
        return K

    @gpflow.params_as_tensors
    def Kdiag(self, X):
        Xp = self.orbit.get_full_orbit(X)
        K = self.basekern.K(Xp)
        return tf.reduce_mean(K, axis=[-2, -1])


class StochasticInvariant(InvariantBase):
    """
    Invariant kernel which returns unbiased estimates of kernel matrices
    """

    @gpflow.decors.params_as_tensors
    def K(self, X, X2=None):
        Xp = self.orbit.get_orbit(X)
        Xp = tf.reshape(Xp, (-1, self.basekern.input_dim))
        Xp2 = tf.reshape(self.orbit.get_orbit(X2), (-1, self.basekern.input_dim)) if X2 is not None else None

        bigK = self.basekern.K(Xp, Xp2)
        bigK_shape = [tf.shape(X)[0], self.orbit.orbit_batch_size, -1, self.orbit.orbit_batch_size]
        bigK = tf.reshape(bigK, bigK_shape)

        if self.orbit.orbit_batch_size < self.orbit.orbit_size:
            bigKt = tf.transpose(bigK, (0, 2, 1, 3))  # N x N2 x M x M
            diag_sum = tf.reduce_sum(tf.matrix_diag_part(bigKt), 2)
            edge_sum = tf.reduce_sum(bigKt, (2, 3)) - diag_sum

            if self.orbit.orbit_size < float("inf"):
                return edge_sum * self.w_edge + diag_sum * self.w_diag
            else:
                return edge_sum / (self.orbit.orbit_batch_size * (self.orbit.orbit_batch_size - 1))
        elif self.orbit.orbit_batch_size == self.orbit.orbit_size:
            return tf.reduce_mean(bigK, [1, 3])

    @gpflow.params_as_tensors
    def Kdiag(self, X):
        Xp = self.orbit.get_orbit(X)

        K = self.basekern.K(Xp)  # [..., C, C]
        axis = [-2, -1]
        if self.orbit.orbit_batch_size < self.orbit.orbit_size:
            diag_sum = tf.reduce_sum(tf.matrix_diag_part(K), axis=-1)
            edge_sum = tf.reduce_sum(K, axis=axis) - diag_sum
            return edge_sum * self.w_edge + diag_sum * self.w_diag
        elif self.orbit.orbit_batch_size == self.orbit.orbit_size:
            return tf.reduce_mean(K, axis=axis)
        raise RuntimeError("Orbit size ({}) must be <= than orbit batch size ({})."
                           .format(self.orbit.orbit_size, self.orbit.orbit_batch_size))

    @property
    def w_diag(self):
        size = self.orbit.orbit_size
        batch_size = self.orbit.orbit_batch_size
        return 1.0 / (batch_size * size)

    @property
    def w_edge(self):
        size = self.orbit.orbit_size
        batch_size = self.orbit.orbit_batch_size
        return (1.0 - 1.0 / size) / (batch_size * (batch_size - 1))

    @property
    def mw_diag(self):
        size = self.orbit.orbit_size
        batch_size = self.orbit.orbit_batch_size
        return 1.0 / size - self.mw_full / batch_size

    @property
    def mw_full(self):
        size = self.orbit.orbit_size
        batch_size = self.orbit.orbit_batch_size
        return batch_size / (batch_size - 1) * (1. - 1. / size)
