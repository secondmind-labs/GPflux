import pickle
import time
import uuid
from pathlib import Path

from experiments.shallow_mnist.refreshed_experiments.data_infrastructure import Dataset


class Experiment:
    def __init__(self, name, dataset, dataset_preprocessor, trainer):
        self._name = name
        self.trainer = trainer
        self._dataset = dataset_preprocessor.preprocess(dataset)
        self._data_preprocessor = dataset_preprocessor

    def run(self):
        results_path = Path('/tmp') / Path(str(self) + '-' + str(uuid.uuid4()))
        results = []
        self.trainer.fit(self._dataset)
        self.trainer.store(results_path)
        # model_path = Path(results_path) / str(self)
        # model_path.mkdir(parents=True, exist_ok=True)
        return results

    @property
    def name(self):
        return self._name

    def __str__(self):
        return '{}_{}'.format(self._name, self._dataset.name)


class ExperimentSuite:
    def __init__(self, experiment_list):
        self._experiment_list = experiment_list

    def run(self):
        for experiment in self._experiment_list:
            experiment.run()


class Trainer:

    def fit(self, dataset: Dataset):
        raise NotImplementedError()

    def store(self, path: Path):
        raise NotImplementedError()


class KerasNNTrainer(Trainer):
    def __init__(self, nn_creator, config):
        self._config = config
        self._nn_creator = nn_creator

    def fit(self, dataset):
        model = self._nn_creator(dataset, self._config)
        init_time = time.time()
        results = model.fit(dataset.train_features, dataset.train_targets,
                            validation_data=(dataset.test_features, dataset.test_targets),
                            epochs=self._config.num_updates // self._config.batch_size,
                            batch_size=self._config.batch_size)

        self._model = model
        self._results = results
        self._duration = time.time() - init_time

    def store(self, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        model_path = path / Path('mymodel.h5')
        self._model.save(str(model_path))
        summary_path = path / Path('summary.txt')
        summary_path.write_text(self._config.summary())
        with summary_path.open(mode='a') as f_handle:
            f_handle.write('\nThe experiemnt took {} min\n'.format(self._duration / 60))
        results_path = path / Path('results.pickle')
        pickle.dump(self._results, open(str(results_path), mode='wb'))


class GPTrainer(Trainer):
    def __init__(self, gp_creator, config):
        self._gp_creator = gp_creator
        self._config = config

    def fit(self, dataset):
        model = self._gp_creator(dataset, self._config)
        init_time = time.time()
        model.compile()
        optimizer = self._config.optimiser
        optimizer.minimize(model, maxiter=self._config.iterations)

        self._model = model
        self._duration = time.time() - init_time

    def store(self):
        save_gpflow_model(filename, self._model)