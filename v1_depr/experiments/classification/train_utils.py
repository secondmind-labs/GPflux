from typing import Callable

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.training import monitor as mon


class ScalarsToTensorBoardTask(mon.BaseTensorBoardTask):
    """
    Monitoring task that creates a TensorBoard with a single scalar value computed by a user
    provided function.
    """

    def __init__(self, file_writer: mon.LogdirWriter, func: Callable) -> None:
        super().__init__(file_writer)
        self.func = func

    def run(self, context: mon.MonitorContext, *args, **kwargs) -> None:
        metrics = self.func(*args, **kwargs)
        summary_values = [tf.Summary.Value(tag=k, simple_value=v)
                          for k, v in metrics.items()]
        summary = tf.Summary(value=summary_values)
        self._file_writer.add_summary(summary, context.iteration_no)
        self._file_writer.flush()


def make_optimize_operation(model, optimizer, session):
    opt = optimizer.optimizer
    objective = model.objective
    variables = model.trainable_tensors

    old_variables = set(opt.variables())
    new_opt_op = opt.minimize(objective, var_list=variables)
    new_variables = set(opt.variables())
    variables_to_init = new_variables - old_variables

    session.run(tf.variables_initializer(list(variables_to_init)))
    return new_opt_op


def make_learning_rate_cb(optimizer):
    """Creates learning rate callback for monitor"""
    lr_scheme = optimizer._optimizer._lr
    sess = gpflow.get_default_session()
    def learning_rate_cb(*args, **kwargs):
        return sess.run(lr_scheme) if type(lr_scheme) is not float else lr_scheme
    return learning_rate_cb


def make_metrics_cb(model, data, suffix, batch_size, num_classes, is_onehot=False):
    """Creates monitor's ELBO and classification error callbacks."""
    y_classes = num_classes
    x, y = data
    num = x.shape[0]
    x_tensor, y_tensor = model.X.parameter_tensor, model.Y.parameter_tensor

    sess = gpflow.get_default_session()
    num_batches = int(np.ceil(num / batch_size))

    elbo = model.likelihood_tensor
    pred_f_mean, pred_f_var = model._build_predict(x_tensor)
    pred_y_mean, pred_y_var = model.likelihood.predict_mean_and_var(
        pred_f_mean, pred_f_var)

    classes_arr = range(y_classes)

    def metrics_cb(*args, **kwargs):
        lml, acc, nlpp = [0.] * 3
        for i in range(num_batches):
            s = i * batch_size
            f = (i + 1) * batch_size
            x_mb = x[s:f, ...]
            y_mb = y[s:f, ...]

            # onehot or class indices
            y_indices_mb = y_mb
            if is_onehot:
                y_indices_mb = np.argmax(y_mb, axis=1).reshape(-1, 1)

            feed_dict = {x_tensor: x_mb, y_tensor: y_mb}
            mb_lml, my_pred, vy_pred = sess.run([elbo, pred_y_mean, pred_y_var],
                                                feed_dict=feed_dict)
            pred_p = my_pred[y_indices_mb == classes_arr]
            pred_p[np.logical_not(np.isfinite(pred_p)) | (pred_p == 0.)] = 1e-10
            nlpp += np.sum(np.log(pred_p))
            lml += mb_lml * len(x_mb)
            acc += np.sum(my_pred.argmax(1) == y_indices_mb[:, 0])

        nlpp /= -num
        lml /= num
        err = (1. - acc / num) * 100
        lml_name = 'lml_' + suffix
        err_name = 'err_' + suffix
        nlpp_name = 'nlpp_' + suffix
        return {lml_name: lml, err_name: err, nlpp_name: nlpp}
    return metrics_cb