import numpy as np

from gpflow.features import InducingPoints
from gpflow.kernels import RBF, Polynomial, White
from gpflow.likelihoods import Gaussian, MultiClass, SoftMax

from gpflux.invariance import Rotation, StochasticInvariant


def RBFPolynomial(dim, degree=0.9):
    return RBF(dim) + Polynomial(dim, degree=degree) + White(dim, 1e-1)


_orbits = {
    'rot': (Rotation, dict(orbit_batch_size=10))
}


_kernels = {
    'rbf': (RBF, dict(variance=10., lengthscales=4.)),
    'poly': (Polynomial, dict(degree=9.0)),
    'rbfpoly': (RBFPolynomial, dict(degree=9.0)),
    'sirot': (StochasticInvariant, dict(basekern='rbf', orbit='rot'))
}


_likelihoods = {
    'gaussian': (Gaussian, dict(variance=0.5)),
    'multiclass': (MultiClass, dict(num_classes=10)),
    'softmax': (SoftMax, dict(num_classes=10)),
}


_features = {
    'ip': (InducingPoints, dict()),
}


def mix_args(default_args, upd_args):
    args = default_args.copy()
    args.update(upd_args)
    return args


def get_object(obj, storage):
    obj_name, obj_new_args = (obj, dict()) if isinstance(obj, str) else obj
    obj_class, obj_default_args = storage[obj_name]
    obj_args = mix_args(obj_default_args, obj_new_args)
    return obj_class, obj_args


def get_likelihood(likelihood):
    lik_class, lik_args = get_object(likelihood, _likelihoods)
    return lik_class(**lik_args)


def get_feature(feature, values, num_points, seed=0):
    feat_class, feat_args = get_object(feature, _features)
    rnd = np.random.RandomState(seed + 1)

    # Select random subset of training data to initialise inducing points with
    # TODO: Generalise to other inducing variables
    values_num = values.shape[0]
    I_idx = rnd.choice(range(values_num), size=num_points, replace=False)
    selected_values = values[I_idx, ...]
    return feat_class(selected_values, **feat_args)


def get_kernel(kernel):
    kern_class, kern_args = get_object(kernel, _kernels)
    basekern_key = 'basekern'
    orbit_key = 'orbit'
    if basekern_key in kern_args:
        kern_args[basekern_key] = get_kernel(kern_args[basekern_key])
    if orbit_key in kern_args:
        kern_args[orbit_key] = get_orbit(kern_args[orbit_key])
    return kern_class(**kern_args)


def get_orbit(orbit):
    orbit_class, orbit_args = get_object(orbit, _orbits)
    return orbit_class(**orbit_args)