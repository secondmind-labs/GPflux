import numpy as np
import tensorflow as tf
from gpflow import settings
from .transformer import spatial_transformer_network as stn


def rotate_img_angles(Ximgs, angles, interpolation_method):
    """
    :param Ximgs: Images to rotate.
    :param angles: Angles in degrees to rotate by.
    :param interpolation_method: Interpolation method.
    :return:
    """
    Ximgs_f32 = tf.cast(Ximgs, tf.float32)

    def rotate(angle):
        # anglerad = tf.cast(angle / 180 * np.pi, settings.float_type)
        anglerad = tf.cast(angle / 180 * np.pi, tf.float32)
        shape = tf.shape(Ximgs)
        return tf.reshape(
            tf.contrib.image.rotate(Ximgs_f32[:, :, :, None], anglerad, interpolation_method),
            (shape[0], shape[1] * shape[2])
        )

    return tf.cast(tf.transpose(tf.map_fn(rotate, angles, dtype=tf.float32), (1, 0, 2)), settings.float_type)


def rotate_img_angles_stn(Ximgs, angles):
    """
    Uses spatial transformer networks to rotate a batch of images by different angles. The entire batch is rotated
    by all angles
    :param Ximgs: input images [None, H, W] or [None, H, W, 1]
    :param angles: angles in degrees to rotate by [P]
    :return: [None, P, H*W]
    """

    if len(Ximgs.get_shape()) == 3:
        Ximgs = tf.expand_dims(Ximgs, -1)  # [?, H, W, 1]

    Ximgs = tf.cast(Ximgs, tf.float32)

    def rotate(angle):
        # Prepare angle
        angle_rad = tf.cast(angle / 180 * np.pi, settings.float_type)
        # Compute affine transformation (tile as per image)
        theta = tf.stack([tf.cos(angle_rad), -tf.sin(angle_rad), 0., tf.sin(angle_rad), tf.cos(angle_rad), 0.])
        theta = tf.reshape(theta, [1, -1])
        theta = tf.cast(tf.tile(theta, [tf.shape(Ximgs)[0], 1]), tf.float32)

        return tf.cast(tf.reshape(
            tf.squeeze(stn(Ximgs, theta)), [tf.shape(Ximgs)[0], tf.shape(Ximgs)[1] * tf.shape(Ximgs)[2]]
        ), settings.float_type)  # [?, H*W]

    return tf.transpose(tf.map_fn(rotate, angles, dtype=settings.float_type), (1, 0, 2))  # [?, P, H*W]


def _apply_stn(Ximgs, theta):
    """
    Use spatial transformer networks to apply a general affine transformation (6 parameters). All images are transformed
    by the SAME theta
    :param Ximgs: [None, H, W, 1]
    :param theta: [6]
    :return: [None, H, W, 1]
    """
    theta = tf.reshape(theta, [1, -1])
    theta = tf.tile(theta, [tf.shape(Ximgs)[0], 1])

    return tf.reshape(tf.squeeze(stn(Ximgs, theta)), [tf.shape(Ximgs)[0], tf.shape(Ximgs)[1] * tf.shape(Ximgs)[2]])


def apply_stn_batch(Ximgs, thetas):
    """
    Use spatial transformer networks to apply a general affine transformation (6 parameters). Every image is transformed
    with every theta
    :param Ximgs: input images [None, H, W] or [None, H, W, 1]
    :param thetas: parameters of the affine transformation by [P, 6]
    :return: [None, P, H*W]
    """

    if len(Ximgs.get_shape()) == 3:
        Ximgs = tf.expand_dims(Ximgs, -1)  # [None, H, W, 1]

    return tf.transpose(tf.map_fn(lambda x: _apply_stn(Ximgs, x), thetas, dtype=settings.float_type), (1, 0, 2))


def softmax_multid(target, axis, name=None):
    """
    Multi dimensional softmax,
    refer to https://github.com/tensorflow/tensorflow/issues/210
    compute softmax along the dimension of target
    the native softmax only supports batch_size x dimension

    https://gist.github.com/raingo/a5808fe356b8da031837
    """
    with tf.name_scope(name, 'softmax', values=[target]):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target - max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / normalize
        return softmax
