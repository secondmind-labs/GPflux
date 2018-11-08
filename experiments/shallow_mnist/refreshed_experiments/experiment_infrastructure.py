import pickle
import time
import uuid
import abc
from collections import namedtuple
from pathlib import Path
from typing import Callable, Any, Type, cast
import tqdm

import gpflow
import gpflow.training.monitor as mon
from experiments.shallow_mnist.refreshed_experiments.conv_gp.configs import GPConfig
from experiments.shallow_mnist.refreshed_experiments.data_infrastructure import Dataset, \
    DatasetPreprocessor
from experiments.shallow_mnist.refreshed_experiments.nn.configs import NNConfig
from experiments.shallow_mnist.refreshed_experiments.utils import Configuration, reshape_to_2d, \
    labels_onehot_to_int, calc_multiclass_error


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
        learning_history = model.fit(dataset.train_features, dataset.train_targets,
                                     validation_data=(dataset.test_features, dataset.test_targets),
                                     batch_size=self.config.batch_size,
                                     epochs=self.config.epochs)

        duration = time.time() - init_time
        return KerasNNTrainer.training_summary(learning_history, model, duration)

    def store(self, training_summary: Trainer.training_summary, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        model_path = path / Path('saved_model.h5')
        training_summary.model.save(str(model_path))
        summary_path = path / Path('summary.txt')
        summary_path.write_text(self._config.summary())
        with summary_path.open(mode='a') as f_handle:
            f_handle.write('\nThe experiemnt took {} min\n'.format(training_summary.duration / 60))


class ClassificationGPTrainer(Trainer):

    @property
    def config(self) -> GPConfig:
        return cast(GPConfig, self._config)

    def fit(self, dataset: Dataset):
        session = gpflow.get_default_session()
        step = mon.create_global_step(session)
        model = self._model_creator(dataset, self.config)
        init_time = time.time()
        model.compile()
        optimiser = self.config.get_optimiser(step)

        x_train, y_train = reshape_to_2d(dataset.train_features), \
                           labels_onehot_to_int(reshape_to_2d(dataset.train_targets))

        x_test, y_test = reshape_to_2d(dataset.test_features), \
                         labels_onehot_to_int(reshape_to_2d(dataset.test_targets))

        opt = optimiser.make_optimize_action(model,
                                             session=session,
                                             var_list=model.trainable_tensors)

        session = model.enquire_session(session)
        train_acc, test_acc = [], []
        num_epochs = self.config.num_epochs
        num_batches = x_train.shape[0] // self.config.batch_size
        with session.as_default():
            for epoch in range(num_epochs):
                for _ in tqdm.tqdm(range(num_batches), desc='Epoch {}'.format(epoch)):
                    opt()
                train_er = calc_multiclass_error(model, x_train[:500, :], y_train[:500, :])
                test_er = calc_multiclass_error(model, x_test, y_test)
                print('Epochs {} train error rate {} test error rate {}.'.format(epoch,
                                                                                 train_er,
                                                                                 test_er))
                train_acc.append((epoch, 1 - train_er))
                test_acc.append((epoch, 1 - test_er))

        duration = time.time() - init_time
        learning_history = {'train_acc': train_acc, 'test_acc': test_acc}
        return Trainer.training_summary(learning_history, model, duration)

    def store(self, training_summary: Trainer.training_summary, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        summary_path = path / Path('summary.txt')
        summary_path.write_text(self._config.summary())
        with summary_path.open(mode='a') as f_handle:
            f_handle.write('\nThe experiemnt took {} min\n'.format(training_summary.duration / 60))
        training_summary_path = path / Path('training_summary.c')
        with training_summary_path.open(mode='wb') as f_handle:
            pickle.dump(training_summary.learning_history, f_handle)
