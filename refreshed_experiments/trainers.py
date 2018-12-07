# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import abc
import pickle
import time
from collections import namedtuple
from pathlib import Path
from typing import cast, Callable, Any
import numpy as np
import gpflow
import tqdm
import tensorflow as tf
from gpflow.session_manager import _DefaultSessionKeeper
from gpflow.training import monitor as mon
from gpflow import misc
from sklearn.model_selection import train_test_split

from refreshed_experiments.configs import NNConfig, GPConfig, _Configuration
from refreshed_experiments.data_infrastructure import Dataset
from refreshed_experiments.utils import reshape_to_2d, labels_onehot_to_int, calc_multiclass_error, \
    calc_avg_nll, save_gpflow_model, get_avg_nll_missclassified, get_top_n_error, name_to_summary, \
    calc_ece_from_probs, calculate_ece_score

TrainingSummary = namedtuple('training_summary',
                             'learning_history model duration')


class Trainer(abc.ABC):

    def __init__(self, model_creator: Callable[[Dataset, _Configuration], Any],
                 config: _Configuration):
        self._config = config
        self._model_creator = model_creator

    @abc.abstractmethod
    def fit(self, dataset: Dataset, path: Path) -> TrainingSummary:
        raise NotImplementedError()

    @abc.abstractmethod
    def store(self, name, training_summary: TrainingSummary, path: Path) -> None:
        raise NotImplementedError()

    @property
    def name(self):
        return self.__class__.__name__


class KerasClassificator(Trainer):

    @property
    def config(self) -> NNConfig:
        return cast(NNConfig, self._config)

    def fit(self, dataset: Dataset, path: Path) -> TrainingSummary:
        model = self._model_creator(dataset, self.config)
        init_time = time.time()
        x, y = dataset.train_features, dataset.train_targets
        x_train, x_valid, y_train, y_valid = \
            train_test_split(x, y, test_size=self.config.validation_proportion)
        summary = model.fit(x_train, y_train,
                            validation_data=(x_valid, y_valid),
                            batch_size=self.config.batch_size,
                            epochs=self.config.epochs,
                            callbacks=self.config.callbacks)
        results_dict = summary.history
        results_dict.update(self._augment_results(model, dataset, results_dict))
        duration = time.time() - init_time
        return TrainingSummary(results_dict, model, duration)

    @staticmethod
    def _augment_results(model, dataset, results_dict):
        test_loss, test_error_top1, test_error_top2, test_error_top3 = model.evaluate(
            dataset.test_features,
            dataset.test_targets)
        print('Final test loss {}, final test error rate {}'.format(test_loss, test_error_top1))
        test_probs = model.predict(dataset.test_features)
        incorrectly_classified = test_probs.argmax(axis=-1) != dataset.test_targets.argmax(axis=-1)
        final_test_avg_nll_missclassified = -np.log(
            test_probs[incorrectly_classified] + 1e-8).mean()
        ece_score = calc_ece_from_probs(test_probs, dataset.test_targets.argmax(axis=-1))
        additional_stats = {'final_error': results_dict['top1_error'][-1],
                            'final_error_top2': results_dict['top2_error'][-1],
                            'final_error_top3': results_dict['top3_error'][-1],
                            'final_test_error': test_error_top1,
                            'final_test_error_top2': test_error_top2,
                            'final_test_error_top3': test_error_top3,
                            'final_test_loss_missclassified': final_test_avg_nll_missclassified,
                            'final_loss': results_dict['loss'][-1],
                            'final_test_loss': test_loss,
                            'final_test_ece': ece_score}
        return additional_stats

    def store(self, name, training_summary: TrainingSummary, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        model_path = path / Path('saved_model.h5')
        training_summary.model.save(str(model_path))
        summary_path = path / Path('summary.txt')
        summary_path.write_text(name_to_summary(name) + self._config.summary())
        with summary_path.open(mode='a') as f_handle:
            f_handle.write('The experiment took {} min\n'.format(training_summary.duration / 60))
        training_summary_path = path / Path('training_summary.c')
        with training_summary_path.open(mode='wb') as f_handle:
            pickle.dump(training_summary.learning_history, f_handle)


class GPClassificator(Trainer):

    @property
    def config(self) -> GPConfig:
        return cast(GPConfig, self._config)

    def fit(self, dataset: Dataset, path: Path):
        init_time = time.time()
        x_train, y_train = reshape_to_2d(dataset.train_features), \
                           labels_onehot_to_int(reshape_to_2d(dataset.train_targets))
        x_test, y_test = reshape_to_2d(dataset.test_features), \
                         labels_onehot_to_int(reshape_to_2d(dataset.test_targets))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(0)
        session = tf.Session(config=config)
        _DefaultSessionKeeper.session = session
        step = mon.create_global_step(session)
        model = self._model_creator(dataset, self.config)
        model.compile()
        optimizer = tf.train.AdamOptimizer()
        all_vars = list(set(model.trainable_tensors))
        all_vars = sorted(all_vars, key=lambda x: x.name)
        # Create optimizer variables before initialization.
        with session.as_default():
            opt = optimizer.minimize(model.objective, var_list=all_vars, )
            model.initialize(session=session)
            misc.initialize_variables(all_vars, session=session, force=False)
        session.run(tf.variables_initializer(optimizer.variables()))

        train_errors_list, test_errors_list, train_avg_nll_list, test_avg_nll_list = [], [], [], []
        num_epochs = self.config.num_epochs
        num_batches = x_train.shape[0] // self.config.batch_size
        stats_fraction = self.config.monitor_stats_num
        monitor = mon.Monitor(self.config.get_tasks(x_test, y_test, model, path), session, step,
                              print_summary=False)

        with monitor:
            with session.as_default():
                for epoch in range(num_epochs):
                    train_error = calc_multiclass_error(model, x_train[:stats_fraction, :],
                                                        y_train[:stats_fraction, :])
                    test_error = calc_multiclass_error(model, x_test[:stats_fraction, :],
                                                       y_test[:stats_fraction, :])
                    train_avg_nll = calc_avg_nll(model, x_train[:stats_fraction, :],
                                                 y_train[:stats_fraction, :])
                    test_avg_nll = calc_avg_nll(model, x_test[:stats_fraction, :],
                                                y_test[:stats_fraction, :])
                    print('Epochs {} train error {} '
                          'test error {} train loss {} test loss {}.'.format(epoch,
                                                                             train_error,
                                                                             test_error,
                                                                             train_avg_nll,
                                                                             test_avg_nll))
                    for _ in tqdm.tqdm(range(num_batches), desc='Epoch {}'.format(epoch)):
                        session.run(opt)
                        monitor(step)

                    train_errors_list.append(train_error)
                    test_errors_list.append(test_error)
                    train_avg_nll_list.append(test_avg_nll)
                    test_avg_nll_list.append(test_avg_nll)
        final_statistics = self.get_final_statistics(model, x_train, y_train, x_test, y_test)
        duration = time.time() - init_time
        learning_history = {'errors': train_errors_list,
                            'val_errors': test_errors_list,
                            'loss': train_avg_nll_list,
                            'test_loss': test_avg_nll_list}
        learning_history.update(final_statistics)
        return TrainingSummary(learning_history, model, duration)

    def store(self, name, training_summary: TrainingSummary, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        summary_path = path / Path('summary.txt')
        summary_path.write_text(name_to_summary(name) + self._config.summary())
        with summary_path.open(mode='a') as f_handle:
            f_handle.write('The experiment took {} min\n'.format(training_summary.duration / 60))
        training_summary_path = path / Path('training_summary.c')
        with training_summary_path.open(mode='wb') as f_handle:
            pickle.dump(training_summary.learning_history, f_handle)
        training_summary_path = path / Path('model.gpflow')
        save_gpflow_model(str(training_summary_path), training_summary.model)

    @staticmethod
    def get_final_statistics(model, x_train, y_train, x_test, y_test):
        final_train_error = calc_multiclass_error(model, x_train, y_train)
        final_test_error = calc_multiclass_error(model, x_test, y_test)
        final_train_avg_nll = calc_avg_nll(model, x_train, y_train)
        final_test_avg_nll = calc_avg_nll(model, x_train, y_train)
        final_test_avg_nll_missclassified = get_avg_nll_missclassified(model, x_test, y_test)
        final_test_acc_top_2_error = get_top_n_error(model, x_test, y_test, n=2)
        final_test_acc_top_3_error = get_top_n_error(model, x_test, y_test, n=3)
        final_train_acc_top_2_error = get_top_n_error(model, x_train, y_train, n=2)
        final_train_acc_top_3_error = get_top_n_error(model, x_train, y_train, n=3)
        final_test_ece = calculate_ece_score(model, x_test, y_test)

        print('Final test loss {}, final test error rate {}'.format(final_test_avg_nll,
                                                                    final_test_error))

        return {'final_error': final_train_error,
                'final_test_error': final_test_error,
                'final_loss': final_train_avg_nll,
                'final_test_loss': final_test_avg_nll,
                'final_test_loss_missclassified': final_test_avg_nll_missclassified,
                'final_test_error_top2': final_test_acc_top_2_error,
                'final_test_error_top3': final_test_acc_top_3_error,
                'final_error_top2': final_train_acc_top_2_error,
                'final_error_top3': final_train_acc_top_3_error,
                'final_test_ece': final_test_ece}
