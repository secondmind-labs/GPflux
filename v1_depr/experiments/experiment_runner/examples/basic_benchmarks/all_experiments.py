# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import argparse

from experiments.experiment_runner.configs import ConvGPConfig, KerasConfig, TickConvGPConfig
from experiments.experiment_runner.creators import BasicCNN, ShallowConvGP
from experiments.experiment_runner.datasets import fashion_mnist, grey_cifar10, mnist, svhn
from experiments.experiment_runner.learners import GPClassificator, KerasClassificator
from experiments.experiment_runner.run_multiple import ExperimentSpecification, run_multiple


def main():
    parser = argparse.ArgumentParser(
        description="""Entrypoint for running multiple experiments.""")

    parser.add_argument('--path', '-p',
                        help='Path to store the saved_results.',
                        type=str,
                        required=True)
    parser.add_argument('--repetitions', '-r',
                        help='Number of repetitions of the experiments.',
                        type=int,
                        default=1)
    parser.add_argument('--gpus',
                        help='Path to store the saved_results.',
                        nargs='+',
                        type=str,
                        required=True)

    basic_set = [mnist, grey_cifar10, fashion_mnist, svhn]

    NN_EPOCH_DICT = {mnist: 12, grey_cifar10: 30, fashion_mnist: 15, svhn: 30}

    args = parser.parse_args()
    experiment_name = 'all_experiments'
    experiments_lists = []
    for dataset in basic_set:
        experiments_lists.append(
            ExperimentSpecification(
                name=experiment_name,
                creator=BasicCNN,
                dataset=dataset,
                config=KerasConfig(num_epochs=NN_EPOCH_DICT[dataset]),
                learner=KerasClassificator))

        experiments_lists.append(
            ExperimentSpecification(
                name=experiment_name,
                creator=ShallowConvGP,
                dataset=dataset,
                config=TickConvGPConfig(),
                learner=GPClassificator),
        )
        experiments_lists.append(
            ExperimentSpecification(
                name=experiment_name,
                creator=ShallowConvGP,
                dataset=dataset,
                config=ConvGPConfig(),
                learner=GPClassificator),
        )

    run_multiple(experiments_lists * args.repetitions, gpus=args.gpus, path=args.path)


if __name__ == '__main__':
    main()
