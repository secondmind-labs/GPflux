# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

"""
Entrypoint for running experiments.
"""

import argparse

from refreshed_experiments.refactored.datasets import grey_cifar10_100epc
from refreshed_experiments.refactored.configs import TickConvGPConfig
from refreshed_experiments.refactored.creators import ShallowConvGP
from refreshed_experiments.refactored.learners import GPClassificator
from refreshed_experiments.refactored.run_multiple import ExperimentSpecification, run_multiple


class TestConfig(TickConvGPConfig):
    def __init__(self):
        super().__init__()
        self.num_epochs = 20


def main():
    parser = argparse.ArgumentParser(
        description="""Entrypoint for running multiple experiments experiments.""")

    parser.add_argument('--path', '-p',
                        help='Path to store the results.',
                        type=str,
                        required=True)

    parser.add_argument('--gpus',
                        help='Path to store the results.',
                        nargs='+',
                        type=str,
                        required=True)

    args = parser.parse_args()

    # specify the list of experiments to run.
    experiments_lists = [
                            ExperimentSpecification(creator=ShallowConvGP,
                                                    dataset=grey_cifar10_100epc,
                                                    config=TickConvGPConfig,
                                                    learner=GPClassificator),
                        ] * 20
    run_multiple(experiments_lists, gpus=args.gpus, path=args.path)


if __name__ == '__main__':
    main()
