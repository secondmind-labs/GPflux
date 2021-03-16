r"""
Callback that enables GPflow's `gpflow.monitor.ModelToTensorBoard` to
integrate with Keras's `tf.keras.Model`\ 's `fit
<https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit>`_ method.
"""
import re
from typing import Any, Dict, Mapping, Optional, Sequence

import tensorflow as tf

import gpflow
from gpflow.utilities import parameter_dict

__all__ = ["TensorBoard"]


class TensorBoard(tf.keras.callbacks.TensorBoard):
    """
    Thin wrapper around Keras' TensorBoard callback that also
    calls GPflow's `gpflow.monitor.ModelToTensorBoard` monitoring task.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        *,
        keywords_to_monitor: Sequence[str] = ("kernel", "mean_function", "likelihood"),
        max_size: int = 3,
        histogram_freq: int = 0,
        write_graph: bool = True,
        write_images: bool = False,
        update_freq: str = "epoch",
        profile_batch: int = 2,
        embeddings_freq: int = 0,
        embeddings_metadata: Optional[Dict] = None,
    ):
        """
        :param log_dir: the path of the directory where to save the log files to be
            parsed by TensorBoard.
        :param max_size: maximum size of arrays (incl.) to store each
            element of the array independently as a scalar in the TensorBoard.
        :param keywords_to_monitor: specifies keywords to be monitored.
            If the parameter's name includes any of the keywords specified it
            will be monitored.

        See [TF's TensorBoard callback docs](
        https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard)
        for docstring of other arguments.
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

    def set_model(self, model: tf.keras.Model) -> None:
        super().set_model(model)
        self.monitor = KerasModelToTensorBoard(
            # mimic Keras' tensorboard callback,
            # which writes all the summaries to "log_dir/train".
            log_dir=self.log_dir + "/train",
            model=self.model,
            max_size=self.max_size,
            keywords_to_monitor=list(self.keywords_to_monitor),
            left_strip_character="._",
        )

    def on_train_batch_end(self, batch: int, logs: Optional[Mapping] = None) -> None:
        """
        Writes scalar summaries for model parameters on every training batch.

        :param batch: index of batch within the current epoch.
        :param logs: metric results for this batch.
        """
        super().on_train_batch_end(batch, logs=logs)

        if self.update_freq == "epoch":
            # We only write at epoch ends
            return

        if batch % self.update_freq == 0:
            self.monitor(batch)

    def on_epoch_end(self, epoch: int, logs: Optional[Mapping] = None) -> None:
        """Run model parameter monitoring at epoch end."""
        super().on_epoch_end(epoch, logs=logs)

        if self.update_freq == "epoch":
            self.monitor(epoch)


class KerasModelToTensorBoard(gpflow.monitor.ModelToTensorBoard):
    """
    Overwrites the run method to deduplicate parameters in parameter_dict.
    """

    # Keras automatically saves all layers in `self._layers[\d]` attribute of the model.
    _LAYER_PARAMETER_REGEXP = re.compile(r"\._layers\[\d+\]\.")

    def _parameter_of_interest(self, match: str) -> bool:
        return self._LAYER_PARAMETER_REGEXP.match(match) is not None

    def run(self, **unused_kwargs: Any) -> None:

        for name, parameter in parameter_dict(self.model).items():
            if not self._parameter_of_interest(name):
                # skip parameters
                continue
            # check if the parameter name matches any of the specified keywords
            if self.summarize_all or any((keyword in name) for keyword in self.keywords_to_monitor):
                # keys are sometimes prepended with a character, which we strip
                name = name.lstrip(self.left_strip_character)
                self._summarize_parameter(name, parameter)
