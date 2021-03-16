# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
"""Utility layer that tracks variables in :class:`tf.Module`."""


import itertools
from functools import wraps
from typing import Any, Callable, Optional, Sequence

import tensorflow as tf


def extend_and_filter(
    extend_method: Callable[..., Sequence],
    filter_method: Optional[Callable[..., Sequence]] = None,
) -> Callable[[Any], Any]:
    """
    Decorator that extends and optionally filters the output of a function.
    Both ``extend_method`` and ``filter_method`` need to be members of the same
    class as the decorated method.

    :param extend_method:
        Accepts the same arguments as the decorated function.
        The returned list from ``extend_method`` will be added to the
        decorated function's returned list.
    :param filter_method:
        Takes in the extended list and filters it.
        Defaults to no filtering when set to `None`.
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapped(self, *args, **kwargs):  # type: ignore
            ret = f(self, *args, **kwargs)
            ret.extend(extend_method(self, *args, **kwargs))
            ret = filter_method(self, ret) if filter_method is not None else ret
            return ret

        return wrapped

    return decorator


class TrackableLayer(tf.keras.layers.Layer):
    r"""
    A :class:`tf.Layer` that tracks variables in :class:`tf.Module`\ s.

    .. todo:: Once TensorFlow 2.5 is released, this class will be removed.
        See https://github.com/Prowler-io/gpflux/issues/189
    """

    @property
    def _submodules(self) -> Sequence[tf.Module]:
        """
        :return: list of :class:`tf.Module` instances that are attributes on the class.
        This also includes instances within lists or tuples.
        """

        submodules = []

        def get_nested_submodules(*objs: Any) -> None:
            for o in objs:
                if isinstance(o, tf.Module):
                    submodules.append(o)

        for obj in self.__dict__.values():
            if isinstance(obj, tf.Module):
                submodules.append(obj)
            elif isinstance(obj, (list, tuple)):
                tf.nest.map_structure(get_nested_submodules, obj)
            elif isinstance(obj, (dict,)):
                tf.nest.map_structure(get_nested_submodules, obj.values())

        return list(dict.fromkeys(submodules))  # remove duplicates, maintaining order (dict 3.6)

    def submodule_variables(self) -> Sequence[tf.Variable]:
        """Return flat iterable of variables from the attributes that are tf.Modules"""
        return list(itertools.chain(*[module.variables for module in self._submodules]))

    def submodule_trainable_variables(self) -> Sequence[tf.Variable]:
        """Return flat iterable of trainable variables from attributes that are tf.Modules"""
        return list(itertools.chain(*[module.trainable_variables for module in self._submodules]))

    def submodule_non_trainable_variables(self) -> Sequence[tf.Variable]:
        """Return flat iterable of non trainable variables from attributes that are tf.Modules"""
        return [v for module in self._submodules for v in module.variables if not v.trainable]

    def _dedup_weights(self, weights):  # type: ignore
        """Deduplicate weights while maintaining order as much as possible."""
        # copy this method from the super class
        # to have it in the local class' namespace
        return super()._dedup_weights(weights)

    @property  # type: ignore
    @extend_and_filter(submodule_trainable_variables, _dedup_weights)
    def trainable_weights(self) -> Sequence[tf.Variable]:
        return super().trainable_weights

    @property  # type: ignore
    @extend_and_filter(submodule_non_trainable_variables, _dedup_weights)
    def non_trainable_weights(self) -> Sequence[tf.Variable]:
        return super().non_trainable_weights

    @property  # type: ignore
    @extend_and_filter(submodule_trainable_variables, _dedup_weights)
    def trainable_variables(self) -> Sequence[tf.Variable]:
        return super().trainable_variables

    @property  # type: ignore
    @extend_and_filter(submodule_variables, _dedup_weights)
    def variables(self) -> Sequence[tf.Variable]:
        return super().variables
