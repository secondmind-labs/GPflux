from abc import ABC, abstractmethod
from functools import lru_cache
from itertools import permutations
from math import factorial

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import settings
from gpflow import transforms

from .transformations import rotate_img_angles, rotate_img_angles_stn, apply_stn_batch

logger = settings.logger()


class Orbit(gpflow.Parameterized, ABC):
    def __init__(self, orbit_batch_size=None, **kwargs):
        """
        :param orbit_batch_size: Number of elements that the orbit is to be approximated with.
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.orbit_batch_size = orbit_batch_size

    def get_orbit(self, X):
        if self.orbit_batch_size is None:
            return self.get_full_orbit(X)

        logger.debug("`get_orbit()` called by `%s` is the default. Probably not the most efficient. " % str(type(self)))
        return tf.transpose(
            tf.random_shuffle(
                tf.transpose(self.get_full_orbit(X), [1, 0, 2])
            )[:self.orbit_batch_size, :, :],
            [1, 0, 2])

    def get_full_orbit(self, X):
        if self.orbit_size == np.inf:
            raise gpflow.GPflowError("Orbit has infinite size, can not return the full orbit.")
        else:
            raise NotImplementedError

    @property
    @abstractmethod
    def orbit_size(self):
        """
        :return: The size of the full orbit.
        """
        pass

    @gpflow.autoflow((settings.float_type,))
    def compute_orbit(self, X):
        return self.get_orbit(X)

    @gpflow.autoflow((settings.float_type,))
    def compute_full_orbit(self, X):
        return self.get_full_orbit(X)


class FlipInputDims(Orbit):
    """
    Kernel for 2D input, invariant to flipping the dimensions. I.e.:
      f([1, 0]) == f([0, 1])
    """

    def get_full_orbit(self, X):
        return tf.stack([X, tf.manip.reverse(X, axis=[1])], axis=1)

    @property
    def orbit_size(self):
        return 2


class Permutation(Orbit):
    def get_full_orbit(self, X):
        perms = tf.constant(np.array(list(permutations(range(self._parent.input_dim)))).flatten()[:, None])
        return tf.reshape(tf.transpose(tf.gather_nd(tf.transpose(X), perms)),
                          (-1, self.orbit_size, self._parent.input_dim))

    @property
    def orbit_size(self):
        return factorial(self._parent.input_dim)


class ImageOrbit(Orbit, ABC):
    @property
    def img_size(self):
        input_dim = self._parent.input_dim
        img_size = [int(input_dim ** 0.5), int(input_dim ** 0.5)]
        assert np.prod(img_size) == input_dim
        return img_size

    @property
    def input_dim(self):
        return self._parent.input_dim


class Rot90(ImageOrbit):
    """
    Rot90
    Kernel invariant to 90 degree rotations of the input image.
    """

    def get_full_orbit(self, X):
        Ximgs = tf.reshape(X, [-1] + self.img_size)
        cc90 = tf.reshape(tf.transpose(tf.reverse(Ximgs, [-1]), [0, 2, 1]), (-1, self.input_dim))
        cc180 = tf.reshape(tf.reverse(Ximgs, [-2, -1]), (-1, np.prod(self.img_size)))
        cc270 = tf.reshape(tf.reverse(tf.transpose(Ximgs, [0, 2, 1]), [-1]), (-1, self.input_dim))
        return tf.concat((X[:, None, :], cc90[:, None, :], cc180[:, None, :], cc270[:, None, :]), 1)

    @property
    def orbit_size(self):
        return 4


class QuantRotation(ImageOrbit):
    """
    QuantRotation
    Kernel invariant to any quantised rotations of the input image.
    """

    def __init__(self, orbit_batch_size=None, rotation_quantisation=45, interpolation_method="NEAREST",
                 same_minibatch=True, **kwargs):
        super().__init__(orbit_batch_size=orbit_batch_size, **kwargs)

        self.rotation_quantisation = rotation_quantisation
        self.interpolation_method = interpolation_method
        assert 360 % rotation_quantisation == 0, "Orbit must complete in 360 degrees."  # Not strictly necessary
        self.angles = np.arange(0, 360, rotation_quantisation)
        self.same_minibatch = same_minibatch

    def get_full_orbit(self, X):
        Ximgs = tf.reshape(X, [-1] + self.img_size)
        return rotate_img_angles(Ximgs, self.angles, self.interpolation_method)

    @lru_cache(maxsize=None)
    def _cached_get_orbit(self, X):
        angles = tf.random_shuffle(self.angles)[:self.orbit_batch_size]
        Ximgs = tf.reshape(X, [-1] + self.img_size)
        return rotate_img_angles(Ximgs, angles, self.interpolation_method)

    def get_orbit(self, X):
        if self.orbit_batch_size is None:
            return self.get_full_orbit(X)

        # We need to explicitly memoize the _tensor_ output here. We need to use the same orbit minibatch for all
        # computations.
        if not self.same_minibatch:
            angles = tf.random_shuffle(self.angles)[:self.orbit_batch_size]
            Ximgs = tf.reshape(X, [-1] + self.img_size)
            return rotate_img_angles(Ximgs, angles, self.interpolation_method)
        else:
            return self._cached_get_orbit(X)

    @property
    def orbit_size(self):
        return len(self.angles)


class Rotation(ImageOrbit):
    def __init__(self, orbit_batch_size=None, angle=179.99, interpolation_method="NEAREST", same_minibatch=True,
                 use_stn=False, **kwargs):
        assert orbit_batch_size is not None
        assert same_minibatch
        super().__init__(orbit_batch_size=orbit_batch_size, **kwargs)
        self.interpolation_method = interpolation_method
        self.angle = gpflow.Param(angle, transform=gpflow.transforms.Logistic(0., 180.))  # constrained to [0, 180]
        self.use_stn = use_stn

    @lru_cache(maxsize=None)
    @gpflow.params_as_tensors
    def get_orbit(self, X):
        # Reparameterise angle
        eps = tf.random_uniform([self.orbit_batch_size], 0., 1., dtype=settings.float_type)
        angles = -self.angle + 2. * self.angle * eps

        Ximgs = tf.reshape(X, [-1] + self.img_size)
        if self.use_stn:
            return rotate_img_angles_stn(Ximgs, angles)  # STN always uses bilinear interpolation
        else:
            return rotate_img_angles(Ximgs, angles, self.interpolation_method)

    @property
    def orbit_size(self):
        return np.inf


class ParameterisedSTN(ImageOrbit):
    """
    Kernel invariant to to transformations using Spatial Transformer Networks (STNs); this corresponds to six-parameter
    affine transformations.
    This version of the kernel uses interpretable parameters:
        - rotation angle
        - scale (in x and y)
        - shear (in x and y)
    """

    def __init__(self, angle, scalex=1., scaley=1., taux=0., tauy=0., orbit_batch_size=None):
        """
        :param orbit_batch_size:
        :param angle: angle in degrees; identity = 0
        :param scalex: scale in x; sample from [0, scalex]; identity = 1
        :param scaley: scale in y; sample from [0, scaley]; identity = 1
        :param taux: shear in x; sample from [-taux, taux]; identity = 0
        :param tauy: shear in y; sample from [-tauy, tauy]; identity = 0
        """
        super().__init__(orbit_batch_size=orbit_batch_size)
        self.angle = gpflow.Param(angle, transform=gpflow.transforms.Logistic(0., 180.))  # constrained to [0, 180]
        self.scalex = gpflow.Param(scalex)  # negative scale = reflection + scale
        self.scaley = gpflow.Param(scaley)
        self.taux = gpflow.Param(taux, transform=transforms.positive)
        self.tauy = gpflow.Param(tauy, transform=transforms.positive)

    @staticmethod
    def _stn_theta_vec(angle_deg, sx, sy, tx, ty):
        """
        Compute 6-parameter theta vector from physical components
        :param angle_deg: rotation angle
        :param sx: scale in x direction
        :param sy: scale in y direction
        :param tx: shear in x direction
        :param ty: shear in y direction
        :return:
        """
        angle_rad = tf.cast(angle_deg / 180 * np.pi, settings.float_type)
        s = tf.sin(angle_rad)
        c = tf.cos(angle_rad)

        return [sx * c - ty * sx * s,
                tx * sy * c - sy * s,
                tf.zeros_like(s),
                sx * s + ty * sx * c,
                tx * sy * s + sy * c,
                tf.zeros_like(s)]

    @lru_cache(maxsize=None)
    @gpflow.params_as_tensors
    def get_orbit(self, X):
        uniform = tf.random_uniform([5, self.orbit_batch_size], 0., 1., dtype=settings.float_type)
        eps_angle = uniform[0]
        eps_scalex = uniform[1]
        eps_scaley = uniform[2]
        eps_taux = uniform[3]
        eps_tauy = uniform[4]
        angles = -self.angle + 2. * self.angle * eps_angle
        scalex = self.scalex * eps_scalex
        scaley = self.scaley * eps_scaley
        taux = -self.taux + 2 * self.taux * eps_taux
        tauy = -self.tauy + 2 * self.tauy * eps_tauy

        thetas = self._stn_theta_vec(angles, scalex, scaley, taux, tauy)
        Ximgs = tf.reshape(X, [-1] + self.img_size)
        return apply_stn_batch(Ximgs, thetas)

    @property
    def orbit_size(self):
        return np.inf



class GeneralSTN(ImageOrbit):
    """
    Kernel invariant to to transformations using Spatial Transformer Networks (STNs); this correponds to six-parameter
    affine transformations.
    This version of the kernel is parameterised by the six independent parameters directly (thus "_general")
    """

    def __init__(self, orbit_batch_size=None,
                 theta_min=np.array([1., 0., 0., 0., 1., 0.]),
                 theta_max=np.array([1., 0., 0., 0., 1., 0.]),
                 constrain=False):
        """
        :param orbit_batch_size:
        :param theta_min: one end of the range; identity = [1, 0, 0, 0, 1, 0]
        :param theta_max: other end of the range; identity = [1, 0, 0, 0, 1, 0]
        :param constrain: whether theta_min is always below the identity and theta_max always above
        """
        super().__init__(orbit_batch_size=orbit_batch_size)
        self.constrain = constrain

        def param(value):
            return gpflow.Param(value, dtype=settings.float_type, transform=transforms.positive)

        if constrain:
            self.theta_min_0 = param(1. - theta_min[0])
            self.theta_min_1 = param(-theta_min[1])
            self.theta_min_2 = param(-theta_min[2])
            self.theta_min_3 = param(-theta_min[3])
            self.theta_min_4 = param(1. - theta_min[4])
            self.theta_min_5 = param(-theta_min[5])

            self.theta_max_0 = gpflow.Param(theta_min[0], dtype=settings.float_type,
                                            transform=transforms.Log1pe(lower=1.))
            self.theta_max_1 = param(theta_min[1])
            self.theta_max_2 = param(theta_min[2])
            self.theta_max_3 = param(theta_min[3])
            self.theta_max_4 = gpflow.Param(theta_min[4], dtype=settings.float_type,
                                            transform=transforms.Log1pe(lower=1.))
            self.theta_max_5 = gpflow.Param(theta_min[5], dtype=settings.float_type, transform=transforms.positive)
        else:
            self.theta_min = gpflow.Param(theta_min, dtype=settings.float_type)
            self.theta_max = gpflow.Param(theta_max, dtype=settings.float_type)

    @lru_cache(maxsize=None)
    @gpflow.params_as_tensors
    def get_orbit(self, X):
        eps = tf.random_uniform([self.orbit_batch_size, 6], 0., 1., dtype=settings.float_type)
        if self.constrain:
            theta_min = tf.stack([1. - self.theta_min_0,
                                  -self.theta_min_1,
                                  -self.theta_min_2,
                                  -self.theta_min_3,
                                  1. - self.theta_min_4,
                                  -self.theta_min_5])
            theta_max = tf.stack([self.theta_max_0, self.theta_max_1, self.theta_max_2, self.theta_max_3,
                                  self.theta_max_4, self.theta_max_5])
            theta_min = tf.reshape(theta_min, [1, -1])
            theta_max = tf.reshape(theta_max, [1, -1])
        else:
            theta_min = tf.reshape(self.theta_min, [1, -1])
            theta_max = tf.reshape(self.theta_max, [1, -1])

        thetas = theta_min + (theta_max - theta_min) * eps
        Ximgs = tf.reshape(X, [-1] + self.img_size)

        return apply_stn_batch(Ximgs, thetas)

    @property
    def orbit_size(self):
        return np.inf


class LocalTransformation(ImageOrbit):
    """
    Kernel invariant to to transformations of local transformations as defined in Loosli2007, Section 3.2.4
    For an "orbit" the images are created using the following update rule:

        x_new = squash_fn( x_old + alpha_x * fx * tx + alpha_y * fy * ty + beta * sqrt(tx**2 + ty**2) )

    where tx/ty are tangent vectors (computed either by convolution with a sobel filter or derivative of a Gaussian) and
    fx/fy denote the local deformations fields. alpha_x/alpha_y correspond to the scales and the beta-term corresponds
    to a thickening/thinning operation.

    Deformation fields are drawn at random at each pixel value and local correlation is introduced by convolving with a
    Gaussian filter of width sigma_d

    Moreover, if alpha_rot/alpha_scale are not None, explicit deformation fields for linearised rotations/scalings
    are included.

    The quash_fn ensures that the image values are in the original range [0, 1]

    """

    def __init__(self, alphax, alphay,
                 beta=0,
                 sigmat=0.9,
                 sigmad=5.,
                 alpha_rot=None,
                 alpha_scale=None,
                 interpret_as_ranges=False,
                 gauss_or_sobel='gauss',
                 squash_fn=None,
                 orbit_batch_size=None):
        super().__init__(orbit_batch_size=orbit_batch_size)
        self.alphax = gpflow.Param(alphax)
        self.alphay = gpflow.Param(alphay)
        if alpha_rot is not None:
            self.alpha_rot = gpflow.Param(alpha_rot)
        else:
            self.alpha_rot = None
        if alpha_scale is not None:
            self.alpha_scale = gpflow.Param(alpha_scale)
        else:
            self.alpha_scale = None
        self.beta = gpflow.Param(beta)
        self.sigmat = gpflow.Param(sigmat, transform=gpflow.transforms.positive)
        self.sigmad = gpflow.Param(sigmad, transform=gpflow.transforms.positive)
        self.interpret_as_ranges = interpret_as_ranges
        self.gauss_or_sobel = gauss_or_sobel
        self.squash_fn = squash_fn
        """
        :param orbit_batch_size:
        :param alphax: scale of x-deformations
        :param alphay: scale of y-deformations
        :param alpha_rot: scale of rotations
        :param alpha_scale: scale of scaling operations
        :param beta: scale of thickening/thinning deformations
        :param sigmat: width of the Gaussian used to filter the image (to compute tx/ty) [only used if gauss_or_sobel=='gauss')
        :param sigmad: correlation length of the deformation field.
        :param interpret_as_ranges: whether to interpret alpha/beta as values are ranges to sample from
        :param gauss_or_sobel: how to compute tx/ty
        :param squash_fn: which squashing function to use (e.g. lambda x: tf.clip_by_value(x, 0., 1.))
        """

    @lru_cache(maxsize=None)
    @gpflow.params_as_tensors
    def get_orbit(self, X):
        """
        Compute orbit of a batch of images X; for each image a different orbit is sampled
        :param X: batch of images [?, H*W]
        :return: orbits for the entire batch [?, orbit_size, H*W]
        """

        def get_orbit_singleim(im):
            """
            Compute orbit of a single image
            :param im: single image [H, W]
            :return: orbit of the single image [P, H*W]
            """
            if self.interpret_as_ranges:
                # draw independent parameters for each image in the orbit
                alphax = tf.random_uniform([self.orbit_batch_size, 1, 1]) * self.alphax
                alphay = tf.random_uniform([self.orbit_batch_size, 1, 1]) * self.alphay
                beta = -self.beta + tf.random_uniform([self.orbit_batch_size, 1, 1]) * 2 * self.beta
                if self.alpha_rot is not None:
                    alpha_rot = -self.alpha_rot + tf.random_uniform([self.orbit_batch_size, 1, 1]) * 2 * self.alpha_rot
                else:
                    alpha_rot = None
                if self.alpha_scale is not None:
                    alpha_scale = -self.alpha_scale + \
                                  tf.random_uniform([self.orbit_batch_size, 1, 1]) * 2 * self.alpha_scale
                else:
                    alpha_scale = None
            else:
                # use same parameter for each orbit
                alphax = self.alphax
                alphay = self.alphay
                beta = self.beta
                alpha_rot = self.alpha_rot
                alpha_scale = self.alpha_scale
            im = tf.tile(tf.expand_dims(im, 0), [self.orbit_batch_size, 1, 1])
            fx, fy = random_deformation_fields(bs=self.orbit_batch_size, sigma=self.sigmad)
            im_filtered = apply_deformation_fields(im, fx, fy, alphax, alphay, beta, self.sigmat,
                                                   alpha_rot=alpha_rot, alpha_scale=alpha_scale,
                                                   clip_fn=self.squash_fn,
                                                   return_only_filtered=True, gauss_or_sobel=self.gauss_or_sobel)
            im_filtered = tf.reshape(im_filtered, [self.orbit_batch_size, -1])  # [P, H*W]
            return im_filtered

        Ximgs = tf.reshape(X, [-1] + self.img_size)  # [?, H, W]

        return tf.map_fn(get_orbit_singleim, Ximgs, dtype=settings.float_type)  # [?, P, H*W]

    @property
    def orbit_size(self):
        return np.inf
        
