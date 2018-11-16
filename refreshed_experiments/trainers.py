import abc
import pickle
import time
from collections.__init__ import namedtuple
from pathlib import Path
from typing import cast, Callable, Any

import gpflow
import keras
import tqdm
from gpflow.training import monitor as mon
from sklearn.model_selection import train_test_split

from refreshed_experiments.configs import NNConfig, GPConfig, Configuration
from refreshed_experiments.data_infrastructure import Dataset
from refreshed_experiments.utils import reshape_to_2d, labels_onehot_to_int, calc_multiclass_error, \
    calc_avg_nll, save_gpflow_model, get_avg_nll_missclassified, get_top_n_error


class Trainer(abc.ABC):
    # we can easily extend the persistent outcome of an experiment
    training_summary = namedtuple('training_summary',
                                  'learning_history model duration')

    def __init__(self, model_creator: Callable[[Dataset, Configuration], Any],
                 config: Configuration):
        self._config = config
        self._model_creator = model_creator

    @abc.abstractmethod
    def fit(self, dataset: Dataset) -> 'Trainer.training_summary':
        raise NotImplementedError()

    @abc.abstractmethod
    def store(self, name, training_summary: 'Trainer.training_summary', path: Path) -> None:
        raise NotImplementedError()

    @property
    def name(self):
        return self.__class__.__name__


class KerasClassificationTrainer(Trainer):

    @property
    def config(self) -> NNConfig:
        return cast(NNConfig, self._config)

    def fit(self, dataset: Dataset) -> Trainer.training_summary:
        model = self._model_creator(dataset, self.config)
        init_time = time.time()
        x, y = dataset.train_features, dataset.train_targets
        x_train, x_valid, y_train, y_valid = \
            train_test_split(x, y, test_size=self.config.validation_proportion)

        callbacks = []
        if self.config.early_stopping:
            callbacks += [keras.callbacks.EarlyStopping(patience=5)]

        summary = model.fit(x_train, y_train,
                            validation_data=(x_valid, y_valid),
                            batch_size=self.config.batch_size,
                            epochs=self.config.epochs,
                            callbacks=callbacks)

        results_dict = summary.history
        test_loss, test_acc = model.evaluate(dataset.test_features, dataset.test_targets)

        duration = time.time() - init_time
        print('Final test loss {}, final test accuracy {}'.format(test_loss, test_acc))
        results_dict.update({'final_acc': results_dict['acc'][-1],
                             'final_test_acc': test_acc,
                             'final_loss': results_dict['loss'][-1],
                             'final_test_loss': test_loss})
        return KerasClassificationTrainer.training_summary(results_dict, model, duration)

    def store(self, name, training_summary: Trainer.training_summary, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        model_path = path / Path('saved_model.h5')
        training_summary.model.save(str(model_path))
        summary_path = path / Path('summary.txt')
        summary_path.write_text(name+'\n')
        summary_path.write_text(self._config.summary())
        with summary_path.open(mode='a') as f_handle:
            f_handle.write('The experiment took {} min\n'.format(training_summary.duration / 60))
        training_summary_path = path / Path('training_summary.c')
        with training_summary_path.open(mode='wb') as f_handle:
            pickle.dump(training_summary.learning_history, f_handle)


class ClassificationGPTrainer(Trainer):

    @property
    def config(self) -> GPConfig:
        return cast(GPConfig, self._config)

    def fit(self, dataset: Dataset):
        init_time = time.time()
        x_train, y_train = reshape_to_2d(dataset.train_features), \
                           labels_onehot_to_int(reshape_to_2d(dataset.train_targets))
        x_test, y_test = reshape_to_2d(dataset.test_features), \
                         labels_onehot_to_int(reshape_to_2d(dataset.test_targets))
        session = gpflow.get_default_session()
        step = mon.create_global_step(session)
        model = self._model_creator(dataset, self.config)
        model.compile()
        optimizer = self.config.get_optimiser(step)
        opt = optimizer.make_optimize_action(model, session=session, global_step=step)

        train_acc_list, test_acc_list, train_avg_nll_list, test_avg_nll_list = [], [], [], []
        num_epochs = self.config.num_epochs
        num_batches = x_train.shape[0] // self.config.batch_size
        stats_fraction = self.config.monitor_stats_fraction
        monitor = mon.Monitor([], session, step, print_summary=False)

        with monitor:
            with session.as_default():
                for epoch in range(num_epochs):
                    train_acc = 100 - calc_multiclass_error(model, x_train[:stats_fraction, :],
                                                            y_train[:stats_fraction, :])
                    test_acc = 100 - calc_multiclass_error(model, x_test[:stats_fraction, :],
                                                           y_test[:stats_fraction, :])
                    train_avg_nll = calc_avg_nll(model, x_train[:stats_fraction, :],
                                                 y_train[:stats_fraction, :])
                    test_avg_nll = calc_avg_nll(model, x_train[:stats_fraction, :],
                                                y_train[:stats_fraction, :])
                    print('Epochs {} train accuracy {} '
                          'test accuracy {} train loss {} test loss {}.'.format(epoch,
                                                                                train_acc,
                                                                                test_acc,
                                                                                train_avg_nll,
                                                                                test_avg_nll))
                    for _ in tqdm.tqdm(range(num_batches), desc='Epoch {}'.format(epoch)):
                        opt()
                        monitor(step)

                    train_acc_list.append(train_acc)
                    test_acc_list.append(test_acc)
                    train_avg_nll_list.append(test_avg_nll)
                    test_avg_nll_list.append(test_avg_nll)
        final_statistics = self.get_final_statistics(model, x_train, y_train, x_test, y_test)
        duration = time.time() - init_time
        learning_history = {'acc': train_acc_list,
                            'val_acc': test_acc_list,
                            'loss': train_avg_nll_list,
                            'val_loss': test_avg_nll_list}
        learning_history.update(final_statistics)
        return Trainer.training_summary(learning_history, model, duration)

    def store(self, name, training_summary: Trainer.training_summary, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        summary_path = path / Path('summary.txt')
        summary_path.write_text(name+'\n')
        summary_path.write_text(self._config.summary())
        with summary_path.open(mode='a') as f_handle:
            f_handle.write('The experiment took {} min\n'.format(training_summary.duration / 60))
        training_summary_path = path / Path('training_summary.c')
        with training_summary_path.open(mode='wb') as f_handle:
            pickle.dump(training_summary.learning_history, f_handle)
        training_summary_path = path / Path('model.gpflow')
        save_gpflow_model(str(training_summary_path), training_summary.model)

    @staticmethod
    def get_final_statistics(model, x_train, y_train, x_test, y_test):
        x_test = x_test[:1000, :]
        x_train = x_train[:1000, :]
        y_train = y_train[:1000, :]
        y_test = y_test[:1000, :]
        final_train_acc = 100 - calc_multiclass_error(model, x_train, y_train)
        final_test_acc = 100 - calc_multiclass_error(model, x_test, y_test)
        final_train_avg_nll = calc_avg_nll(model, x_train, y_train)
        final_test_avg_nll = calc_avg_nll(model, x_train, y_train)
        final_test_avg_nll_missclassified = get_avg_nll_missclassified(model, x_test, y_test)
        final_test_acc_top_2_acc = 1 - get_top_n_error(model, x_test, y_test, n=2)
        final_test_acc_top_3_acc = 1 - get_top_n_error(model, x_test, y_test, n=3)

        print(
            'Final test loss {}, final test accuracy {}'.format(final_test_avg_nll, final_test_acc))
        print(
            'Final test loss  missclassified {}, final test top 2 {} acc, final test top 3 {} acc'
            ''.format(final_test_avg_nll_missclassified, final_test_acc_top_2_acc,
                      final_test_acc_top_3_acc))

        return {'final_acc': final_train_acc,
                'final_test_acc': final_test_acc,
                'final_loss': final_train_avg_nll,
                'final_test_loss': final_test_avg_nll,
                'final_test_loss_missclassified': final_test_avg_nll_missclassified,
                'final_test_acc_2top': final_test_acc_top_2_acc,
                'final_test_acc_3top': final_test_acc_top_3_acc}
