#
# Copyright (c) 2021 The GPflux Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
r"""
This module implements a callback that enables GPflow's `gpflow.monitor.ModelToTensorBoard` to
integrate with Keras's `tf.keras.Model`\ 's `fit
<https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_ method.
"""
import re
from typing import Any, Dict, List, Mapping, Optional, Union

import gpflow
from gpflow.keras import tf_keras
from gpflow.utilities import parameter_dict

__all__ = ["TensorBoard"]


class TensorBoard(tf_keras.callbacks.TensorBoard):
    """
    This class is a thin wrapper around a `tf.keras.callbacks.TensorBoard` callback that also
    calls GPflow's `gpflow.monitor.ModelToTensorBoard` monitoring task.
    """

    log_dir: str
    """
    The path of the directory to which to save the log files to be
    read by TensorBoard. Files are saved in ``log_dir / "train"``,
    following the Keras convention.
    """

    keywords_to_monitor: List[str]
    """
    Parameters whose name match a keyword in the *keywords_to_monitor* list
    will be added to the TensorBoard.
    """

    update_freq: Union[int, str]
    r"""
    Either an integer or ``"epoch"``. If using an integer *n*, write
    losses/metrics/parameters at every *n*\ th batch; when using
    ``"epoch"``, write them at the end of each epoch. Note that writing too
    frequently to TensorBoard can slow down the training.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        *,
        keywords_to_monitor: List[str] = [
            "kernel",
            "mean_function",
            "likelihood",
        ],
        max_size: int = 3,
        histogram_freq: int = 0,
        write_graph: bool = True,
        write_images: bool = False,
        update_freq: Union[int, str] = "epoch",
        profile_batch: int = 2,
        embeddings_freq: int = 0,
        embeddings_metadata: Optional[Dict] = None,
    ):
        """
        :param log_dir: The path of the directory to which to save the log files to be
            read by TensorBoard.
        :param keywords_to_monitor: A list of keywords to monitor.
            If the parameter's name includes any of the keywords specified, it
            will be added to the TensorBoard.
        :param max_size: The maximum size of arrays (inclusive) for which each element is
            written independently as a scalar to the TensorBoard.

        For information on all other arguments, see TensorFlow's `TensorBoard callback docs
        <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard>`_.
        """

        super().__init__(
            log_dir=log_dir,
            histogram_freq=histogram_freq,
            write_graph=write_graph,
            write_images=write_images,
            update_freq=update_freq,
            profile_batch=profile_batch,
            embeddings_freq=embeddings_freq,
            embeddings_metadata=embeddings_metadata,
        )
        self.keywords_to_monitor = keywords_to_monitor
        self.max_size = max_size

    def set_model(self, model: tf_keras.Model) -> None:
        """
        Set the model (extends the Keras `set_model
        <https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard#set_model>`_
        method).

        This method initialises :class:`KerasModelToTensorBoard` and mimics Keras's TensorBoard
        callback in writing the summary logs to :attr:`log_dir` / "train".
        """
        super().set_model(model)
        self.monitor = KerasModelToTensorBoard(
            log_dir=self.log_dir + "/train",
            model=self.model,
            max_size=self.max_size,
            keywords_to_monitor=list(self.keywords_to_monitor),
            left_strip_character="._",
        )

    def on_train_batch_end(self, batch: int, logs: Optional[Mapping] = None) -> None:
        """
        Write to TensorBoard if :attr:`update_freq` is an integer.

        :param batch: The index of the batch within the current epoch.
        :param logs: The metric results for this batch.
        """
        super().on_train_batch_end(batch, logs=logs)

        if self.update_freq == "epoch":
            # We only write at epoch ends
            return

        if isinstance(self.update_freq, int) and batch % self.update_freq == 0:
            self.monitor(batch)

    def on_epoch_end(self, epoch: int, logs: Optional[Mapping] = None) -> None:
        """Write to TensorBoard if :attr:`update_freq` equals ``"epoch"``."""
        super().on_epoch_end(epoch, logs=logs)

        if self.update_freq == "epoch":
            self.monitor(epoch)


class KerasModelToTensorBoard(gpflow.monitor.ModelToTensorBoard):
    """
    This class overwrites the :meth:`run` method to deduplicate parameters in
    :attr:`parameter_dict`.
    """

    # Keras automatically saves all layers in the `self._self_tracked_trackables`
    # attribute of the model, which is a list of sub-modules.
    _LAYER_PARAMETER_REGEXP = re.compile(r"\._self_tracked_trackables\[\d+\]\.")

    def _parameter_of_interest(self, match: str) -> bool:
        return self._LAYER_PARAMETER_REGEXP.match(match) is not None

    def run(self, **unused_kwargs: Any) -> None:
        """Write the model's parameters to TensorBoard."""

        for name, parameter in parameter_dict(self.model).items():
            if not self._parameter_of_interest(name):
                # skip parameters
                continue
            # check if the parameter name matches any of the specified keywords
            if self.summarize_all or any((keyword in name) for keyword in self.keywords_to_monitor):
                # keys are sometimes prepended with a character, which we strip
                name = name.lstrip(self.left_strip_character)
                self._summarize_parameter(name, parameter)
