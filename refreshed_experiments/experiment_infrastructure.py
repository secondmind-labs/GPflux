# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import uuid
from pathlib import Path

from refreshed_experiments.data_infrastructure import Dataset
from refreshed_experiments.trainers import Trainer, TrainingSummary


class Experiment:
    def __init__(self, name: str,
                 dataset: Dataset,
                 trainer: Trainer):
        self._name = name
        self.trainer = trainer
        self._dataset = dataset
        self._uuid = str(uuid.uuid4())

    def run(self, path: Path) -> TrainingSummary:
        training_summary = self.trainer.fit(self._dataset, path)
        return training_summary

    def store(self, training_summary: TrainingSummary, path: Path):
        self.trainer.store(self._name, training_summary, path)

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        return '{}_{}'.format(self._name, self._uuid)


class ExperimentRunner:

    def __init__(self, experiment_list):
        self._experiment_list = experiment_list

    def run(self, path: Path):
        for experiment in self._experiment_list:
            summary = experiment.run(path / Path(str(experiment)))
            experiment.store(summary, path / Path(str(experiment)))

