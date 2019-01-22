# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import argparse

from experiments.experiment_runner.datasets import random_mnist_10epc, random_mnist_1epc, \
    random_mnist_20epc
from experiments.experiment_runner.configs import TickConvGPConfig, KerasConfig, LongKerasConfig, \
    SmallTickConvGPConfig, KerasConvGPLikeConfig
from experiments.experiment_runner.creators import ShallowConvGP, BasicCNN
from experiments.experiment_runner.learners import GPClassificator, KerasClassificationLearner
from experiments.experiment_runner.run_multiple import ExperimentSpecification, run_multiple


def main():
    parser = argparse.ArgumentParser(
        description="""Entrypoint for running multiple experiments.""")

    parser.add_argument('--path', '-p',
                        help='Path to store the results.',
                        type=str,
                        required=True)
    parser.add_argument('--repetitions', '-r',
                        help='Number of repetitions of the experiments.',
                        type=int,
                        default=1)
    parser.add_argument('--gpus',
                        help='Path to store the results.',
                        nargs='+',
                        type=str,
                        required=True)

    args = parser.parse_args()
    experiment_name = 'mnist_fractions_experiment2'
    experiments_lists = []

    experiments_lists.append(
        ExperimentSpecification(
            name=experiment_name,
            creator=BasicCNN,
            dataset=random_mnist_1epc,
            config=KerasConvGPLikeConfig,
            learner=KerasClassificationLearner))
    experiments_lists.append(
        ExperimentSpecification(
            name=experiment_name,
            creator=BasicCNN,
            dataset=random_mnist_10epc,
            config=KerasConvGPLikeConfig,
            learner=KerasClassificationLearner))
    experiments_lists.append(
        ExperimentSpecification(
            name=experiment_name,
            creator=BasicCNN,
            dataset=random_mnist_20epc,
            config=KerasConvGPLikeConfig,
            learner=KerasClassificationLearner))

    experiments_lists.append(
        ExperimentSpecification(
            name=experiment_name,
            creator=ShallowConvGP,
            dataset=random_mnist_1epc,
            config=SmallTickConvGPConfig,
            learner=GPClassificator))
    experiments_lists.append(
        ExperimentSpecification(
            name=experiment_name,
            creator=ShallowConvGP,
            dataset=random_mnist_10epc,
            config=SmallTickConvGPConfig,
            learner=GPClassificator))
    experiments_lists.append(
        ExperimentSpecification(
            name=experiment_name,
            creator=ShallowConvGP,
            dataset=random_mnist_20epc,
            config=SmallTickConvGPConfig,
            learner=GPClassificator))

    run_multiple(experiments_lists * args.repetitions, gpus=args.gpus, path=args.path)


if __name__ == '__main__':
    main()
