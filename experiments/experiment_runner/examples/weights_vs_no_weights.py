# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential


import argparse
import pickle
from pathlib import Path
from typing import NamedTuple, Any

import gpflow

from experiments.experiment_runner.datasets import random_mnist_10epc, random_mnist_1epc, \
    random_mnist_100epc
from experiments.experiment_runner.configs import TickConvGPConfig, KerasConfig
from experiments.experiment_runner.creators import ShallowConvGP, BasicCNN
from experiments.experiment_runner.learners import GPClassificator, KerasClassificator
from experiments.experiment_runner.results_managing import Summary
from experiments.experiment_runner.run_multiple import ExperimentSpecification, run_multiple
from experiments.experiment_runner.utils import calc_nll_and_error_rate

import numpy as np

_parser = argparse.ArgumentParser(
    description="""Entrypoint for running multiple experiments.""")

_parser.add_argument('--path', '-p',
                     help='Path to store the saved_results.',
                     type=str,
                     required=True)
_parser.add_argument('--repetitions', '-r',
                     help='Number of repetitions of the experiments.',
                     type=int,
                     default=1)
_parser.add_argument('--gpus',
                     help='Path to store the saved_results.',
                     nargs='+',
                     type=str,
                     required=True)

_args = _parser.parse_args()

SmallCustomHistory = NamedTuple('CustomHistory',
                                [
                                    ('test_error_rate_list', Any),
                                    ('test_avg_nll_list', Any),
                                    ('test_predictions', Any),
                                ])

CustomHistory = NamedTuple('CustomHistory', [
    ('test_error_rate_list', Any),
    ('test_avg_nll_list', Any),
    ('train_error_rate_list', Any),
    ('train_avg_nll_list', Any),
    ('test_predictions', Any),
    ('variances', Any),
    ('lenghtscales', Any),
    ('test_variance_f', Any),
    ('train_variance_f', Any)
])


class StatsGatheringGPClassificator(GPClassificator):

    def _initialise_stats_containers(self):
        self._test_error_rate_list = []
        self._test_avg_nll_list = []

        self._train_error_rate_list = []
        self._train_avg_nll_list = []

        self._test_predictions = []
        self._weights = []
        self._variance = []
        self._lengthscale = []
        self._test_variance_f = []
        self._train_variance_f = []
        self._test_features = None
        self._test_labels = None

    def _gather_statistics(self, dataset, config, epoch):
        NUM_F_POINTS = 10
        if config.with_weights:
            w = self._model.layers[0].kern.weights.read_value(gpflow.get_default_session())
        else:
            w = 0
        l = self._model.layers[0].kern.basekern.lengthscales.read_value(
            gpflow.get_default_session())
        v = self._model.layers[0].kern.basekern.variance.read_value(gpflow.get_default_session())

        _, train_var_f = self._model.predict_f(dataset.train_features[:NUM_F_POINTS])
        _, test_var_f = self._model.predict_f(dataset.test_features[:NUM_F_POINTS])

        self._weights.append(w)
        self._lengthscale.append(l)
        self._variance.append(v)
        self._train_variance_f.append(train_var_f)
        self._test_variance_f.append(test_var_f)

        x_train, y_train, x_test, y_test = dataset.train_features, \
                                           dataset.train_targets, \
                                           dataset.test_features, \
                                           dataset.test_targets
        if self._test_features is None:
            self._test_features = x_test[:config.monitor_stats_num, :]
            self._test_labels = y_test[:config.monitor_stats_num, :]

        test_avg_nll, test_error_rate, test_p \
            = calc_nll_and_error_rate(self._model,
                                      x_test[:config.monitor_stats_num, :],
                                      y_test[:config.monitor_stats_num, :])
        train_avg_nll, train_error_rate, _ \
            = calc_nll_and_error_rate(self._model,
                                      x_train[:config.monitor_stats_num, :],
                                      y_train[:config.monitor_stats_num, :])

        print('Epochs {} test error rate {:.3f} test loss {:.3f}.'.format(epoch,
                                                                          test_error_rate,
                                                                          test_avg_nll, ))

        self._test_avg_nll_list.append(test_avg_nll)
        self._test_error_rate_list.append(test_error_rate)
        self._train_avg_nll_list.append(train_avg_nll)
        self._train_error_rate_list.append(train_error_rate)
        self._test_predictions.append(np.take(test_p, y_test[:config.monitor_stats_num, :]))

    def _create_history(self):
        custom_history = dict(
            test_error_rate_list=self._test_error_rate_list,
            test_avg_nll_list=self._test_avg_nll_list,
            test_predictions=self._test_predictions,
            variances=self._variance,
            lenghtscales=self._lengthscale,
            test_variance_f=self._test_variance_f,
            train_variance_f=self._train_variance_f,
            train_error_rate_list=self._train_error_rate_list,
            train_avg_nll_list=self._train_avg_nll_list)
        return custom_history

    def get_summary(self, outcome, data_source):
        return Summary([], [])

    def store(self, outcome, summary, experiment_path, experiment_name):
        training_summary_path = experiment_path / Path('training_summary.c')
        with training_summary_path.open(mode='wb') as f_handle:
            pickle.dump(outcome.history, f_handle)


class StatsGatheringKerasClassificationLearner(KerasClassificator):
    def _initialise_stats_containers(self):
        self._test_error_rate_list = []
        self._test_avg_nll_list = []
        self._predictions = []
        self._test_features = None
        self._test_labels = None

    def _gather_statistics(self, dataset, config, epoch):
        x_train, y_train, x_test, y_test = dataset.train_features, \
                                           dataset.train_targets, \
                                           dataset.test_features, \
                                           dataset.test_targets
        p_test = self._model.predict(dataset.test_features)
        p_train = self._model.predict(dataset.train_features)
        p_test += 1e-12
        p_train += 1e-12
        p_test = p_test / p_test.sum(-1, keepdims=True)
        if self._test_features is None:
            self._test_features = x_test
            self._test_labels = y_test

        test_avg_nll = -np.log((p_test * y_test).sum(-1)).mean()
        test_error_rate = (1 - np.mean(p_test.argmax(-1) == y_test.argmax(-1))) * 100
        print('Epochs {} test error rate {:.3f} test loss {:.3f}.'.format(epoch,
                                                                          test_error_rate,
                                                                          test_avg_nll, ))
        self._test_avg_nll_list.append(test_avg_nll)
        self._test_error_rate_list.append(test_error_rate)
        self._predictions.append((p_test * y_test).sum(-1))

    def _create_history(self):
        custom_history = dict(
            test_predictions=self._predictions,
            test_error_rate_list=self._test_error_rate_list,
            test_avg_nll_list=self._test_avg_nll_list)
        return custom_history

    def get_summary(self, outcome, data_source):
        return Summary(outcome.history, [])

    def store(self, outcome, summary, experiment_path, experiment_name):
        training_summary_path = experiment_path / Path('training_summary.c')
        with training_summary_path.open(mode='wb') as f_handle:
            pickle.dump(outcome.history, f_handle)


class TickConvGPNoWeightsConfig(TickConvGPConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_weights = False


class TickConvGPRegularisedConfig(TickConvGPConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_weights = False


def simple(dataset_list):
    NUM_GP_EPOCHS = 500
    STEPS_PER_EPOCH = 100
    STATS_NUM = 10000
    STORE_FREQ = 100000
    NUM_INDUCING_POINTS = 1000
    keras_config = KerasConfig(epochs=STEPS_PER_EPOCH * NUM_GP_EPOCHS // 10, steps_per_epoch=10)

    gp_config_no_weights = TickConvGPNoWeightsConfig(steps_per_epoch=STEPS_PER_EPOCH,
                                                     num_epochs=NUM_GP_EPOCHS,
                                                     init_patches='patches-unique',
                                                     monitor_stats_num=STATS_NUM,
                                                     patch_shape=[5, 5],
                                                     store_frequency=STORE_FREQ,
                                                     num_inducing_points=NUM_INDUCING_POINTS,
                                                     with_weights=False)

    gp_config_weights = TickConvGPConfig(steps_per_epoch=STEPS_PER_EPOCH,
                                         num_epochs=NUM_GP_EPOCHS,
                                         init_patches='patches-unique',
                                         monitor_stats_num=STATS_NUM,
                                         patch_shape=[5, 5],
                                         store_frequency=STORE_FREQ,
                                         num_inducing_points=NUM_INDUCING_POINTS,
                                         with_weights=True)

    experiment_name = 'example_experiment'
    experiments_lists = []
    for dataset in dataset_list:
        experiments_lists.extend([
            ExperimentSpecification(
                name=experiment_name,
                creator=BasicCNN,
                dataset=dataset.load_data(),
                config=keras_config,
                learner=StatsGatheringKerasClassificationLearner),
            ExperimentSpecification(
                name=experiment_name,
                creator=ShallowConvGP,
                dataset=dataset.load_data(),
                config=gp_config_weights,
                learner=StatsGatheringGPClassificator),
            ExperimentSpecification(
                name=experiment_name,
                creator=ShallowConvGP,
                dataset=dataset.load_data(),
                config=gp_config_no_weights,
                learner=StatsGatheringGPClassificator),
        ])

    run_multiple(experiments_lists * _args.repetitions, gpus=_args.gpus, path=_args.path)


if __name__ == '__main__':
    dataset_list = [random_mnist_1epc, random_mnist_10epc, random_mnist_100epc]
    simple(dataset_list)
