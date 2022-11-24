from functools import wraps
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from tensorflow_probability.python.util.deferred_tensor import TensorMetaClass

from gpflow.base import TensorType


class LikelihoodOutputs(tf.Module, metaclass=TensorMetaClass):
    """
    This class encapsulates the outputs of a :class:`~gpflux.layers.LikelihoodLayer`.

    It contains the mean and variance of the marginal distribution of the final latent
    :class:`~gpflux.layers.GPLayer`, as well as the mean and variance of the likelihood.

    This class includes the `TensorMetaClass
    <https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/util/deferred_tensor.py#L81>`_
    to make objects behave as a `tf.Tensor`. This is necessary so that it can be
    returned from the `tfp.layers.DistributionLambda` Keras layer.
    """

    def __init__(
        self,
        f_mean: TensorType,
        f_var: TensorType,
        y_mean: Optional[TensorType],
        y_var: Optional[TensorType],
    ):
        super().__init__(name="likelihood_outputs")

        self.f_mean = f_mean
        self.f_var = f_var
        self.y_mean = y_mean
        self.y_var = y_var

    def _value(
        self, dtype: tf.dtypes.DType = None, name: str = None, as_ref: bool = False
    ) -> tf.Tensor:
        return self.f_mean

    @property
    def shape(self) -> tf.Tensor:
        return self.f_mean.shape

    @property
    def dtype(self) -> tf.dtypes.DType:
        return self.f_mean.dtype

def batch_predict(
    predict_callable: Callable[[np.ndarray], Tuple[np.ndarray, ...]], batch_size: int = 1000
) -> Callable[[np.ndarray], Tuple[np.ndarray, ...]]:
    """
    Simple wrapper that transform a full dataset predict into batch predict.
    :param predict_callable: desired predict function that we want to wrap so it's executed in
     batch fashion.
    :param batch_size: how many predictions to do within single batch.
    """
    if batch_size <= 0:
        raise ValueError(f"Batch size has to be positive integer!")

    @wraps(predict_callable)
    def wrapper(x: np.ndarray) -> Tuple[np.ndarray, ...]:
        batches_f_mean = []
        batches_f_var = []
        batches_y_mean = []
        batches_y_var = []
        for x_batch in tf.data.Dataset.from_tensor_slices(x).batch(
            batch_size=batch_size, drop_remainder=False
        ):
            batch_predictions = predict_callable(x_batch)
            batches_f_mean.append(batch_predictions.f_mean)
            batches_f_var.append(batch_predictions.f_var)
            batches_y_mean.append(batch_predictions.y_mean)
            batches_y_var.append(batch_predictions.y_var)

        return LikelihoodOutputs(
            tf.concat(batches_f_mean, axis=0),
            tf.concat(batches_f_var, axis=0),
            tf.concat(batches_y_mean, axis=0),
            tf.concat(batches_y_var, axis=0)
        )

    return wrapper






"""
NOTE -- I think these functions are completely redunant now 
"""

def draw_gaussian_at(support, sd=1.0, height=1.0, xpos=0.0, ypos=0.0, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    gaussian = np.exp((-support ** 2.0) / (2 * sd ** 2.0))
    gaussian /= gaussian.max()
    gaussian *= height
    return ax.plot(gaussian + xpos, support + ypos, **kwargs)

def timer(start,end):
       hours, rem = divmod(end-start, 3600)
       minutes, seconds = divmod(rem, 60)
       print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
