# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

"""
Entrypoint for running experiments.

Example usage:
```
python run_experiments.py --gpus 0 1 --path test
```

Note on setting up the experiments:
creator should be implemented in ../creators.py
config should be implemented in ../configs.py
dataset should be implemented in ../data.py
learner should be implemented in ../learners.py
"""

import argparse

from experiments.experiment_runner.datasets import grey_cifar10_100epc, mnist
from experiments.experiment_runner.configs import TickConvGPConfig
from experiments.experiment_runner.creators import ShallowConvGP
from experiments.experiment_runner.learners import GPClassificator
from experiments.experiment_runner.run_multiple import ExperimentSpecification, run_multiple


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
    experiment_name = 'my_experiment'
    experiments_lists = [
        ExperimentSpecification(
            name=experiment_name,
            creator=ShallowConvGP,
            dataset=grey_cifar10_100epc,
            config=TickConvGPConfig,
            learner=GPClassificator),
        ExperimentSpecification(
            name=experiment_name,
            creator=ShallowConvGP,
            dataset=mnist,
            config=TickConvGPConfig,
            learner=GPClassificator),
    ]

    run_multiple(experiments_lists, gpus=args.gpus, path=args.path)


if __name__ == '__main__':
    main()
