import uuid
from pathlib import Path

from refreshed_experiments.data_infrastructure import Dataset
from refreshed_experiments.trainers import Trainer


class Experiment:
    def __init__(self, name: str,
                 dataset: Dataset,
                 trainer: Trainer):
        self._name = name
        self.trainer = trainer
        self._dataset = dataset

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


class ExperimentRunner:

    def __init__(self, experiment_list):
        self._experiment_list = experiment_list

    def run(self, path: Path):
        for experiment in self._experiment_list:
            summary = experiment.run()
            experiment.store(summary, path / Path(str(experiment)))


