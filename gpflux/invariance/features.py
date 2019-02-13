# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import gpflow
import tensorflow as tf
from gpflow.dispatch import dispatch

from .kernels import InvariantBase, Invariant, StochasticInvariant


class InvariantInducingPoints(gpflow.features.InducingPoints):
    pass


class StochasticInvariantInducingPoints(InvariantInducingPoints):
    pass


@dispatch(InvariantInducingPoints, InvariantBase)  # I.e. shared between invariant and non-invariant
def Kuu(feat, kern, *, jitter=0.0):
    """
    Inducing feature placed in the non-invariant base function that the
    invariant GP is constructed from.
    :param feat:
    :param kern:
    :param jitter:
    :return:
    """
    with gpflow.params_as_tensors_for(feat):
        Kzz = kern.basekern.K(feat.Z)
        Kzz += jitter * tf.eye(len(feat), dtype=gpflow.settings.dtypes.float_type)
    return Kzz


@dispatch(InvariantInducingPoints, Invariant, object)  # Non-invariant only
def Kuf(feat, kern, Xnew):
    with gpflow.params_as_tensors_for(feat):
        N, M = tf.shape(Xnew)[0], tf.shape(feat.Z)[0]
        Xorbit = kern.orbit.get_full_orbit(Xnew)  # N x orbit_size x D
        bigKzx = kern.basekern.K(feat.Z, tf.reshape(Xorbit, (N * kern.orbit.orbit_size, -1)))  # M x N * orbit_size
        Kzx = tf.reduce_mean(tf.reshape(bigKzx, (M, N, kern.orbit.orbit_size)), [2])
    return Kzx


@dispatch(StochasticInvariantInducingPoints, StochasticInvariant, object)
def Kuf(feat, kern, Xnew):
    """
    :return: M x N x orbit_batch_size
    """
    with gpflow.params_as_tensors_for(feat):
        N, M = tf.shape(Xnew)[0], tf.shape(feat.Z)[0]
        Xorbit = kern.orbit.get_orbit(Xnew)  # N x orbit_size x D
        Kzx = kern.basekern.K(feat.Z, tf.reshape(Xorbit, (N * kern.orbit.orbit_batch_size, -1)))  # M x N * orbit_size
    return tf.reshape(Kzx, (M, N, kern.orbit.orbit_batch_size))
