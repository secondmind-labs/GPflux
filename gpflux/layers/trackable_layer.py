# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""A base Keras Layer that tracks variables and weights in GPflux"""


from functools import wraps
import itertools

import tensorflow as tf


def extend_with_method(method):
    """"
    This decorator calls a decorated method which returns a list, and extends
    the result with the return value of another method on the same class. This
    method is called after the decorated function, with the same arguments as
    the decorated function.
    """

    def decorator(f):
        @wraps(f)
        def wrapped(self, *args, **kwargs):
            ret = f(self, *args, **kwargs)
            ret.extend(method(self, *args, **kwargs))
            return ret

        return wrapped

    return decorator


class TrackableLayer(tf.keras.layers.Layer):
    """
    A tf.Layer that implements tracking of tf.Variables on the class's
    attributes that are tf.Modules.

    Currently, tf.Modules track the tf.Variables of their attributes that are
    also tf.Modules.  Similarly, tf.Layers track the tf.Variables of their
    attributes that are also tf.Layers.  However, despite the fact that
    tf.Layer inherits from tf.Module, they cannot track the tf.Variables of
    their attributes that are generic tf.Modules. This seems to be an issue
    that the TensorFlow authors seem to want to fix in the future.
    """

    @property
    def _submodules(self):
        """Return a list of tf.Module instances that are attributes on the class"""
        return [v for v in self.__dict__.values() if isinstance(v, tf.Module)]

    def submodule_variables(self):
        """Return flat iterable of variables from the attributes that are tf.Modules"""
        return list(itertools.chain(*[module.variables for module in self._submodules]))

    def submodule_trainable_variables(self):
        """Return flat iterable of trainable variables from attributes that are tf.Modules"""
        return list(
            itertools.chain(
                *[module.trainable_variables for module in self._submodules]
            )
        )

    def submodule_non_trainable_variables(self):
        """Return flat iterable of non trainable variables from attributes that are tf.Modules"""
        return [
            v
            for module in self._submodules
            for v in module.variables
            if not v.trainable
        ]

    @property  # type: ignore
    @extend_with_method(submodule_trainable_variables)
    def trainable_weights(self):
        return super().trainable_weights

    @property  # type: ignore
    @extend_with_method(submodule_non_trainable_variables)
    def non_trainable_weights(self):
        return super().non_trainable_weights

    @property  # type: ignore
    @extend_with_method(submodule_trainable_variables)
    def trainable_variables(self):
        return super().trainable_variables

    @property  # type: ignore
    @extend_with_method(submodule_variables)
    def variables(self):
        return super().variables
