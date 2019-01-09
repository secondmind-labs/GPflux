# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import pickle
import time
from pathlib import Path
from typing import NamedTuple, List

import gpflow
import keras
import tqdm
from gpflow.training import monitor as mon
from sklearn.model_selection import train_test_split

from refreshed_experiments.refactored.core import Learner, LearnerOutcome
from refreshed_experiments.refactored.results_managing import ScalarSequence, Scalar, Summary
from refreshed_experiments.refactored.data import StaticDataSource
from refreshed_experiments.refactored.utils import get_text_summary
from refreshed_experiments.refactored.utils import reshape_to_2d, labels_onehot_to_int, calc_multiclass_error, \
    calc_avg_nll, save_gpflow_model


class KerasLearner(Learner):

    def __init__(self, model: keras.models.Model):
        self._model = model

    def learn(self, data_source: StaticDataSource, config, path):
        dataset = data_source.get_data()
        x, y = dataset.train_features, dataset.train_targets
        x_train, x_valid, y_train, y_valid = \
            train_test_split(x, y, test_size=config.validation_proportion)
        time_init = time.time()
        summary = self._model.fit(x_train, y_train,
                                  validation_data=(x_valid, y_valid),
                                  batch_size=config.batch_size,
                                  epochs=config.epochs,
                                  callbacks=config.callbacks +
                                            [keras.callbacks.TensorBoard(log_dir=path)])
        time_done = time.time()
        return LearnerOutcome(self._model, config, summary.history, time_done - time_init)

    def get_summary(self, outcome, data_source: StaticDataSource):
        dataset = data_source.get_data()
        outcome.model.evaluate(dataset.test_features, dataset.test_targets)
        history = outcome.history
        scalars = [
            Scalar(history['loss'][-1], name='training_loss'),
            Scalar(history['val_loss'][-1], name='validation_loss')
        ]
        x_axis_name = 'optimisation steps'
        scalar_sequences = [
            ScalarSequence(history['loss'],
                           name='training_loss',
                           x_axis_name=x_axis_name,
                           y_axis_name='training_loss'),
            ScalarSequence(history['val_loss'],
                           name='test_loss',
                           x_axis_name=x_axis_name,
                           y_axis_name='test_loss'),
        ]
        return Summary(scalars=scalars,
                       scalar_sequences=scalar_sequences)

    def store(self, outcome: LearnerOutcome, summary, experiment_path, experiment_name):
        model, config = outcome.model, outcome.config
        model_path = experiment_path / Path('saved_model.h5')
        model.save(str(model_path))
        summary_path = experiment_path / Path('summary.txt')
        summary_str = 'Experiment {}\n'.format(experiment_name) + config.summary()
        summary_str += get_text_summary(summary)
        summary_path.write_text(summary_str)
        with summary_path.open(mode='a') as f_handle:
            f_handle.write('The experiment took {} min\n'.format(outcome.duration / 60))
        training_summary_path = experiment_path / Path('training_summary.c')
        with training_summary_path.open(mode='wb') as f_handle:
            pickle.dump(summary, f_handle)


class GPClassificator(Learner):

    def __init__(self, model):
        self._model = model

    def learn(self, data_source, config, path):
        gpflow.reset_default_graph_and_session()
        dataset = data_source.get_data()
        x_train, y_train = reshape_to_2d(dataset.train_features), \
                           labels_onehot_to_int(reshape_to_2d(dataset.train_targets))
        x_test, y_test = reshape_to_2d(dataset.test_features), \
                         labels_onehot_to_int(reshape_to_2d(dataset.test_targets))
        session = gpflow.get_default_session()
        step = mon.create_global_step(session)
        lr = config.get_learning_rate(step)
        self._model.compile()
        optimizer = config.optimiser(lr)
        opt = optimizer.make_optimize_action(self._model)

        train_errors_list, test_errors_list, train_avg_nll_list, test_avg_nll_list = [], [], [], []
        num_epochs = config.num_epochs
        num_batches = x_train.shape[0] // config.batch_size
        stats_fraction = config.monitor_stats_num
        monitor = mon.Monitor(config.get_tasks(x_test, y_test, self._model, path), session, step,
                              print_summary=False)

        time_init = time.time()
        with session.as_default(), monitor:
            for epoch in range(1, num_epochs + 1):
                train_error_rate = calc_multiclass_error(self._model, x_train[:stats_fraction, :],
                                                         y_train[:stats_fraction, :])
                test_error_rate = calc_multiclass_error(self._model, x_test[:stats_fraction, :],
                                                        y_test[:stats_fraction, :])
                train_avg_nll = calc_avg_nll(self._model, x_train[:stats_fraction, :],
                                             y_train[:stats_fraction, :])
                test_avg_nll = calc_avg_nll(self._model, x_test[:stats_fraction, :],
                                            y_test[:stats_fraction, :])
                print('Epochs {} train error rate {} '
                      'test error rate {} train loss {} '
                      'test loss {}.'.format(epoch,
                                             train_error_rate,
                                             test_error_rate,
                                             train_avg_nll,
                                             test_avg_nll))
                for _ in tqdm.tqdm(range(num_batches), desc='Epoch {}'.format(epoch)):
                    opt()
                    monitor(step)

                train_errors_list.append(train_error_rate)
                test_errors_list.append(test_error_rate)
                train_avg_nll_list.append(test_avg_nll)
                test_avg_nll_list.append(test_avg_nll)

        time_done = time.time()
        gp_classification_history = GPClassificationHistory(train_errors_list,
                                                            test_errors_list,
                                                            train_avg_nll_list,
                                                            test_avg_nll_list)

        return LearnerOutcome(self._model, config, gp_classification_history, time_done - time_init)

    def get_summary(self, outcome, data_source: StaticDataSource):
        dataset = data_source.get_data()
        x_train, y_train = reshape_to_2d(dataset.train_features), \
                           labels_onehot_to_int(reshape_to_2d(dataset.train_targets))
        x_test, y_test = reshape_to_2d(dataset.test_features), \
                         labels_onehot_to_int(reshape_to_2d(dataset.test_targets))
        test_error_rate = calc_multiclass_error(self._model, x_test, y_test)
        test_loss = calc_avg_nll(outcome.model, x_test, y_test)
        train_error_rate = calc_multiclass_error(self._model, x_train, y_train)
        train_loss = calc_avg_nll(outcome.model, x_train, y_train)
        scalars = [
            Scalar(test_loss, name='test_loss'),
            Scalar(train_loss, name='train_loss'),
            Scalar(test_error_rate, name='test_error_rate'),
            Scalar(train_error_rate, name='train_error_rate'),
        ]
        x_axis_name = 'optimisation steps'
        scalar_sequences = [
            ScalarSequence(outcome.history.train_avg_nll_list,
                           name='training loss',
                           x_axis_name=x_axis_name,
                           y_axis_name='training_loss'),
            ScalarSequence(outcome.history.test_avg_nll_list,
                           name='test loss',
                           x_axis_name=x_axis_name,
                           y_axis_name='test_loss'),
            ScalarSequence(outcome.history.train_errors_list,
                           name='train error rate',
                           x_axis_name=x_axis_name,
                           y_axis_name='train_error_rate'),
            ScalarSequence(outcome.history.test_errors_list,
                           name='test error rate',
                           x_axis_name=x_axis_name,
                           y_axis_name='test_error_rate_list'),
        ]
        return Summary(scalars=scalars,
                       scalar_sequences=scalar_sequences)

    def store(self, outcome, summary, experiment_path, experiment_name):
        model, config = outcome.model, outcome.config
        model_path = experiment_path / Path('saved_model.h5')
        summary_path = experiment_path / Path('summary.txt')
        summary_str = 'Experiment {}\n'.format(experiment_name) + config.summary()
        summary_str += '\n' + get_text_summary(summary)
        summary_path.write_text(summary_str)
        with summary_path.open(mode='a') as f_handle:
            f_handle.write('The experiment took {} min\n'.format(outcome.duration / 60))
        training_summary_path = experiment_path / Path('training_summary.c')
        with training_summary_path.open(mode='wb') as f_handle:
            pickle.dump(summary, f_handle)
        save_gpflow_model(str(model_path), outcome.model)


GPClassificationHistory = NamedTuple('GPClassificationHistory',
                                     [('train_errors_list', List[float]),
                                      ('test_errors_list', List[float]),
                                      ('train_avg_nll_list', List[float]),
                                      ('test_avg_nll_list', List[float])
                                      ])
