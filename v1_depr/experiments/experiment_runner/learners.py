# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import pickle
import time
from pathlib import Path
from typing import List, NamedTuple

import keras
import numpy as np
import tqdm

import gpflow
from gpflow.training import monitor as mon

from experiments.experiment_runner.core import Learner, LearnerOutcome
from experiments.experiment_runner.data import StaticDataSource
from experiments.experiment_runner.results_managing import Scalar, ScalarSequence, Summary
from experiments.experiment_runner.utils import (
    calc_ece_from_probs,
    calc_nll_and_error_rate,
    calculate_ece_score,
    get_text_summary,
    get_top_n_error,
    reshape_dataset_to2d,
    save_gpflow_model,
    top1_error,
    top2_error,
    top3_error,
)


class KerasClassificator(Learner):

    def __init__(self, model: keras.models.Model):
        self._model = model
        self._initialise_stats_containers()

    def _initialise_stats_containers(self):
        self._train_error_rate_list = []
        self._train_loss_list = []
        self._test_error_rate_list = []
        self._test_loss_list = []

    def learn(self, data_source: StaticDataSource, config, path):
        self._model.compile(loss='categorical_crossentropy',
                            optimizer=config.optimiser(),
                            metrics=[top1_error, top2_error, top3_error])
        dataset = data_source.get_data()
        steps_per_epoch = config.steps_per_epoch if config.steps_per_epoch is not None \
            else max(dataset.train_features.shape[0] // config.batch_size, 1)
        time_init = time.time()
        for epoch in range(config.epochs):
            self._gather_statistics(dataset, config, epoch)
            for _ in range(steps_per_epoch):
                batch_ind = np.random.randint(0, len(dataset.train_features), config.batch_size)
                self._model.train_on_batch(dataset.train_features[batch_ind],
                                           dataset.train_targets[batch_ind])
        time_done = time.time()
        history = self._create_history()
        return LearnerOutcome(self._model, config, history, time_done - time_init)

    def _gather_statistics(self, dataset, config, epoch):
        test_loss, test_error_rate, *_ = \
            self._model.evaluate(dataset.test_features, dataset.test_targets, verbose=0)
        train_loss, train_error_rate, *_ = \
            self._model.evaluate(dataset.train_features, dataset.train_targets, verbose=0)
        self._train_error_rate_list.append(train_error_rate)
        self._test_error_rate_list.append(test_error_rate)
        self._train_loss_list.append(train_loss)
        self._test_loss_list.append(test_loss)
        print('Epochs {} train error rate {:.3f} '
              'test error rate {:.3f} train loss {:.3f} '
              'test loss {:.3f}.'.format(epoch,
                                         train_error_rate,
                                         test_error_rate,
                                         train_loss,
                                         test_loss))

    def _create_history(self):
        return KerasClassificationHistory(train_error_rate_list=self._train_error_rate_list,
                                          test_error_rate_list=self._test_error_rate_list,
                                          train_loss_list=self._train_loss_list,
                                          test_loss_list=self._test_loss_list)

    def get_summary(self, outcome, data_source: StaticDataSource):
        dataset = data_source.get_data()
        test_loss, test_error_rate, top2_error_rate, top3_error_rate \
            = outcome.model.evaluate(dataset.test_features, dataset.test_targets, verbose=0)
        train_loss, train_error_rate, *_ = \
            outcome.model.evaluate(dataset.train_features, dataset.train_targets, verbose=0)
        predicted_y_test = outcome.model.predict(dataset.test_features)
        test_ece_score = calc_ece_from_probs(predicted_y_test, dataset.test_targets.argmax(axis=-1))

        scalars = [
            Scalar(train_loss, name='train loss'),
            Scalar(test_loss, name='test loss'),
            Scalar(test_error_rate, name='test error rate'),
            Scalar(train_error_rate, name='train error rate'),
            Scalar(top2_error_rate, name='test error rate top2'),
            Scalar(top3_error_rate, name='test error rate top3'),
            Scalar(test_ece_score, name='test ece score')
        ]
        x_axis_name = 'optimisation steps [x{}]'.format(outcome.config.steps_per_epoch)
        scalar_sequences = [
            ScalarSequence(outcome.history.train_loss,
                           name='train loss',
                           x_axis_name=x_axis_name,
                           y_axis_name='train loss'),
            ScalarSequence(outcome.history.test_loss,
                           name='test loss',
                           x_axis_name=x_axis_name,
                           y_axis_name='test loss'),
            ScalarSequence(outcome.history.train_error_rate_list,
                           name='train error rate',
                           x_axis_name=x_axis_name,
                           y_axis_name='train error rate'),
            ScalarSequence(outcome.history.test_error_rate_list,
                           name='test error rate',
                           x_axis_name=x_axis_name,
                           y_axis_name='test error rate'),
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


KerasClassificationHistory = NamedTuple('KerasClassificationHistory',
                                        [('train_error_rate_list', List[float]),
                                         ('test_error_rate_list', List[float]),
                                         ('train_loss_list', List[float]),
                                         ('test_loss_list', List[float])
                                         ])


class GPClassificator(Learner):

    def __init__(self, model):
        self._model = model
        self._initialise_stats_containers()

    def learn(self, data_source, config, path):
        gpflow.reset_default_graph_and_session()
        self._model.compile()
        dataset = reshape_dataset_to2d(data_source.get_data())  # gpflux assumes 2d data
        session = gpflow.get_default_session()
        step = mon.create_global_step(session)
        lr = config.get_learning_rate(step)
        optimizer = config.optimiser(lr)
        opt = optimizer.make_optimize_action(self._model, global_step=step)
        monitor = mon.Monitor(monitor_tasks=config.get_tasks(dataset.test_features,
                                                             dataset.test_targets, self._model,
                                                             path, optimizer),
                              session=session,
                              global_step_tensor=step,
                              print_summary=False)

        print('Running for {} epochs, with {} steps per epoch.'.format(config.num_epochs,
                                                                       config.steps_per_epoch))

        time_init = time.time()
        with session.as_default(), monitor:
            for epoch in range(1, config.num_epochs + 1):
                self._gather_statistics(dataset, config, epoch)
                for _ in tqdm.tqdm(range(config.steps_per_epoch), desc='Epoch {}'.format(epoch)):
                    opt()
                    monitor(step)
        time_done = time.time()
        history = self._create_history()
        return LearnerOutcome(model=self._model,
                              config=config,
                              history=history,
                              duration=time_done - time_init)

    def _create_history(self):
        gp_classification_history = GPClassificationHistory(self._train_error_rate_list,
                                                            self._test_error_rate_list,
                                                            self._train_loss,
                                                            self._test_loss)
        return gp_classification_history

    def _initialise_stats_containers(self):
        self._train_error_rate_list = []
        self._train_loss = []
        self._test_error_rate_list = []
        self._test_loss = []

    def _gather_statistics(self, dataset, config, epoch):
        t0 = time.time()
        x_train, y_train, x_test, y_test = dataset.train_features, \
                                           dataset.train_targets, \
                                           dataset.test_features, \
                                           dataset.test_targets
        stats_fraction = config.monitor_stats_num
        train_avg_nll, train_error_rate, _ = calc_nll_and_error_rate(self._model,
                                                                     x_train[:stats_fraction, :],
                                                                     y_train[:stats_fraction, :])
        test_avg_nll, test_error_rate, _ = calc_nll_and_error_rate(self._model,
                                                                   x_test[:stats_fraction, :],
                                                                   y_test[:stats_fraction, :])
        t1 = time.time()
        print('Epochs {} train error rate {:.3f} '
              'test error rate {:.3f} train loss {:.3f} '
              'test loss {:.3f}, testing time {:.3f}.'.format(epoch,
                                                              train_error_rate,
                                                              test_error_rate,
                                                              train_avg_nll,
                                                              test_avg_nll,
                                                              t1 - t0))
        self._train_error_rate_list.append(train_error_rate)
        self._test_error_rate_list.append(test_error_rate)
        self._train_loss.append(train_avg_nll)
        self._test_loss.append(test_avg_nll)

    def get_summary(self, outcome, data_source: StaticDataSource):
        t0 = time.time()
        dataset = reshape_dataset_to2d(data_source.get_data())
        x_train, y_train, x_test, y_test = dataset.train_features, \
                                           dataset.train_targets, \
                                           dataset.test_features, \
                                           dataset.test_targets
        test_error_rate_top2, test_error_rate_top3 = get_top_n_error(self._model, x_test, y_test,
                                                                     n_list=[2, 3])
        ece_score = calculate_ece_score(self._model, x_test, y_test)
        train_loss, train_error_rate, _ = calc_nll_and_error_rate(self._model,
                                                                  x_train,
                                                                  y_train)
        test_loss, test_error_rate, _ = calc_nll_and_error_rate(self._model,
                                                                x_test,
                                                                y_test)

        x_axis_name = 'optimisation steps [x{}]'.format(outcome.config.steps_per_epoch)
        scalars = [
            Scalar(test_loss, name='test loss'),
            Scalar(train_loss, name='train loss'),
            Scalar(train_error_rate, name='train error rate'),
            Scalar(test_error_rate, name='test error rate'),
            Scalar(test_error_rate_top2, name='test error rate top2'),
            Scalar(test_error_rate_top3, name='test error rate top3'),
            Scalar(ece_score, name='test ece score'),
        ]
        scalar_sequences = [
            ScalarSequence(outcome.history.train_avg_nll_list,
                           name='train loss',
                           x_axis_name=x_axis_name,
                           y_axis_name='train loss'),
            ScalarSequence(outcome.history.test_avg_nll_list,
                           name='test loss',
                           x_axis_name=x_axis_name,
                           y_axis_name='test loss'),
            ScalarSequence(outcome.history.train_errors_list,
                           name='train error rate',
                           x_axis_name=x_axis_name,
                           y_axis_name='train error rate'),
            ScalarSequence(outcome.history.test_errors_list,
                           name='test error rate',
                           x_axis_name=x_axis_name,
                           y_axis_name='test error rate'),
        ]
        t1 = time.time()
        print('Getting summary took {:.3f}'.format(t1 - t0))
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
