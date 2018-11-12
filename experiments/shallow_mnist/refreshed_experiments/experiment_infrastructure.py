import pickle
import time
import uuid
import abc
from collections import namedtuple
from pathlib import Path
from typing import Callable, Any, Type, cast
import tqdm
import tensorflow as tf

import gpflow
import gpflow.training.monitor as mon

from experiments.shallow_mnist.refreshed_experiments.conv_gp.configs import GPConfig, ConvGPConfig
from experiments.shallow_mnist.refreshed_experiments.data_infrastructure import Dataset, \
    DatasetPreprocessor
from experiments.shallow_mnist.refreshed_experiments.nn.configs import NNConfig
from experiments.shallow_mnist.refreshed_experiments.utils import Configuration, reshape_to_2d, \
    labels_onehot_to_int, calc_multiclass_error, calc_avg_nll


class Trainer(abc.ABC):
    # we can easily extend the persistent outcome of an experiment
    training_summary = namedtuple('training_summary',
                                  'learning_history model duration')

    def __init__(self, model_creator: Callable[[Dataset, Configuration], Any],
                 config: Type[Configuration]):
        self._config = config
        self._model_creator = model_creator

    @abc.abstractmethod
    def fit(self, dataset: Dataset) -> 'Trainer.training_summary':
        raise NotImplementedError()

    @abc.abstractmethod
    def store(self, training_summary: 'Trainer.training_summary', path: Path) -> None:
        raise NotImplementedError()


class Experiment:
    def __init__(self, name: str,
                 dataset: Dataset,
                 dataset_preprocessor: Type[DatasetPreprocessor],
                 trainer: Trainer):
        self._name = name
        self.trainer = trainer
        self._dataset = dataset_preprocessor.preprocess(dataset)
        self._data_preprocessor = dataset_preprocessor

    def run(self) -> Trainer.training_summary:
        training_summary = self.trainer.fit(self._dataset)
        return training_summary

    def store(self, training_summary: Trainer.training_summary, path: Path):
        self.trainer.store(training_summary, path)

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return '{}_{}'.format(self._name, str(uuid.uuid4()))


class ExperimentSuite:

    def __init__(self, experiment_list):
        self._experiment_list = experiment_list

    def run(self, path: Path):
        for experiment in self._experiment_list:
            summary = experiment.run()
            experiment.store(summary, path / Path(str(experiment)))


class KerasNNTrainer(Trainer):

    @property
    def config(self) -> NNConfig:
        return cast(NNConfig, self._config)

    def fit(self, dataset: Dataset) -> Trainer.training_summary:
        model = self._model_creator(dataset, self.config)
        init_time = time.time()
        summary = model.fit(dataset.train_features, dataset.train_targets,
                            validation_data=(dataset.test_features, dataset.test_targets),
                            batch_size=self.config.batch_size,
                            epochs=self.config.epochs)

        duration = time.time() - init_time
        return KerasNNTrainer.training_summary(summary.history, model, duration)

    def store(self, training_summary: Trainer.training_summary, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        model_path = path / Path('saved_model.h5')
        training_summary.model.save(str(model_path))
        summary_path = path / Path('summary.txt')
        summary_path.write_text(self._config.summary())
        with summary_path.open(mode='a') as f_handle:
            f_handle.write('\nThe experiment took {} min\n'.format(training_summary.duration / 60))
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
        model = self._model_creator(x_train, y_train, self.config)
        model.compile()
        optimizer = ConvGPConfig.get_optimiser(step)
        opt = optimizer.make_optimize_action(model,
                                             session=session,
                                             global_step=step)

        train_acc_list, test_acc_list, train_avg_nll_list, test_avg_nll_list = [], [], [], []
        num_epochs = self.config.num_epochs
        num_batches = x_train.shape[0] // self.config.batch_size
        stats_fraction = self.config.stats_fraction

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

                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                train_avg_nll_list.append(test_avg_nll)
                test_avg_nll_list.append(test_avg_nll)

        final_train_acc = 100 - calc_multiclass_error(model, x_train, y_train)
        final_test_acc = 100 - calc_multiclass_error(model, x_test, y_test)
        final_train_avg_nll = calc_avg_nll(model, x_train, y_train)
        final_test_avg_nll = calc_avg_nll(model, x_train, y_train)

        duration = time.time() - init_time
        learning_history = {'acc': train_acc_list, 'val_acc': test_acc_list,
                            'loss': train_avg_nll_list, 'val_loss': test_avg_nll_list,
                            'final_acc': final_train_acc, 'final_val_acc': final_test_acc,
                            'final_loss': final_train_avg_nll, 'final_val_loss': final_test_avg_nll}
        return Trainer.training_summary(learning_history, model, duration)

    def store(self, training_summary: Trainer.training_summary, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        summary_path = path / Path('summary.txt')
        summary_path.write_text(self._config.summary())
        with summary_path.open(mode='a') as f_handle:
            f_handle.write('\nThe experiment took {} min\n'.format(training_summary.duration / 60))
        training_summary_path = path / Path('training_summary.c')
        with training_summary_path.open(mode='wb') as f_handle:
            pickle.dump(training_summary.learning_history, f_handle)
