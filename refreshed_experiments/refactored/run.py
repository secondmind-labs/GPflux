# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import argparse
from pathlib import Path

from refreshed_experiments.refactored import datasets
from refreshed_experiments.refactored import configs, creators, learners
from refreshed_experiments.refactored.core import Experiment, Trainer
from refreshed_experiments.refactored.data import StaticDataSource
from refreshed_experiments.refactored.utils import get_from_module


def main():
    parser = argparse.ArgumentParser(
        description="""Entrypoint for running the experiments. Run with:
        python run.py -d dataset -mc model_creator -t trainer -c config
        Available are:\n""".format())

    parser.add_argument('--model_creator', '-m',
                        help='Model creator, one of the classes in model_creators.py',
                        type=str, required=True)
    parser.add_argument('--config', '-c', help='Config, one of classes in configs.py',
                        type=str, required=True)
    parser.add_argument('--dataset', '-d', help='Dataset, one of classes in datasets.py.',
                        type=str, required=True)
    parser.add_argument('--learner', '-l', help='Learner class.',
                        type=str, required=True)
    parser.add_argument('--path', '-p', help='The path were results will be stored.', type=Path,
                        required=True)
    parser.add_argument('--repetitions', '-r', help='The number of repetitions of an experiment',
                        type=int, default=1)

    args = parser.parse_args()

    config = get_from_module(args.config, configs)()
    model_creator = get_from_module(args.model_creator, creators)
    learner_class = get_from_module(args.learner, learners)
    dataset = get_from_module(args.dataset, datasets)().load_data()
    path = args.path

    for _ in range(args.repetitions):
        data_source = StaticDataSource(dataset=dataset)
        trainer = Trainer(model_creator, learner_class)
        experiment = Experiment('test_experiment',
                                data_source=data_source,
                                trainer=trainer,
                                config=config)
        experiment.run(path)


if __name__ == '__main__':
    main()
