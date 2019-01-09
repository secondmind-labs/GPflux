# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import abc
from pathlib import Path
from typing import NamedTuple, Any

from refreshed_experiments.refactored.data import DataSource
from refreshed_experiments.refactored.utils import short_uuid


class Learner(abc.ABC):

    def learn(self, data_source: DataSource, config, path: Path):
        pass

    def get_summary(self, outcome, data_source: DataSource):
        pass

    def store(self, outcome, summary, experiment_path,  experiment_name):
        pass

    @property
    def name(self):
        return self.__class__.__name__


LearnerOutcome = NamedTuple('LearnerOutcome',
                            [('model', Any),
                             ('config', Any),
                             ('history', Any),
                             ('duration', float)])


class LearnerCreator:

    def __init__(self, name, create_method):
        self._create_method = create_method
        self._name = name

    def create(self, data_source, config):
        return self._create_method(data_source, config)

    @property
    def name(self):
        return self._name


class Trainer:

    def __init__(self, model_creator, learner_class):
        self._model_creator = model_creator()
        self._learner_class = learner_class
        self._model_name = self._model_creator.name

    def train(self, data_source, config, path, name):
        model = self._model_creator.create(data_source, config)
        learner = self._learner_class(model)
        exp_name = '{}_{}_{}'.format(name, learner.name, short_uuid())
        experiment_path = path / Path(exp_name)
        experiment_path.mkdir(exist_ok=True, parents=True)
        outcome = learner.learn(data_source, config, experiment_path)
        summary = learner.get_summary(outcome, data_source)
        learner.store(outcome, summary, experiment_path, exp_name.split('/')[-1])

    @property
    def model_name(self):
        return self._model_name


class Experiment:
    def __init__(self,
                 name: str,
                 data_source: DataSource,
                 config,
                 trainer: Trainer):
        self._name = name
        self._trainer = trainer
        self._data_source = data_source
        self._config = config

    def run(self, results_path):
        name = '{}/{}/{}_{}'.format(self._name,
                                    self._data_source.name,
                                    self._trainer.model_name,
                                    self._config.name)
        self._trainer.train(self._data_source, self._config, results_path, name)
