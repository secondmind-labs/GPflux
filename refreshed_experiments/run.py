# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import argparse
from pathlib import Path

from refreshed_experiments.experiment_infrastructure import ExperimentRunner, Experiment
from refreshed_experiments.utils import get_from_module
from refreshed_experiments import configs, model_creators, trainers, datasets

"""
Entrypoint for running experiments. Example usage:
```
python run.py -mc convgp_creator -d svhn_5percent -c TickConvGPConfig -t ClassificationGPTrainer -p /tmp/results
```
This will build the model using `convgp_creator` using `TickConvGPConfig` config and run 
`ClassificationGPTrainer` on `svhn_5percent` dataset. The results will be stored to `/tmp/results`. 
Run python run.py --help to see more detailed description. 
"""


def get_name(trainer, config, creator, dataset):
    return "exp-{}-{}-{}-{}".format(trainer.name, config.name, creator.__name__,
                                           dataset.name)


def main():
    parser = argparse.ArgumentParser(
        description="""Entrypoint for running the experiments. Run with:
        python run.py -d dataset -mc model_creator -t trainer -c config
        Available are:\n""".format())

    parser.add_argument('--model_creator', '-mc',
                        help='Model creator, one of the classes in model_creators.py',
                        type=str, required=True)
    parser.add_argument('--config', '-c', help='Config, one of classes in configs.py',
                        type=str, required=True)
    parser.add_argument('--dataset', '-d', help='Dataset, one of classes in datasets.py.',
                        type=str, required=True)
    parser.add_argument('--trainer', '-t', help='Trainer, one of the trainers in trainers.py.',
                        type=str, required=True)
    parser.add_argument('--path', '-p', help='The path were results will be stored.', type=Path,
                        required=True)
    parser.add_argument('--repetitions', '-r', help='The number of repetitions of an experiment',
                        type=int, default=1)

    args = parser.parse_args()

    config = get_from_module(args.config, configs)()
    model_creator = get_from_module(args.model_creator, model_creators)
    trainer = get_from_module(args.trainer, trainers)
    dataset = get_from_module(args.dataset, datasets)().load_data()

    for _ in range(args.repetitions):
        trainer_instance = trainer(model_creator=model_creator,
                                   config=config)
        name = get_name(trainer_instance, config, model_creator, dataset)
        print('Running {}'.format(name))
        experiment = Experiment(name=name,
                                dataset=dataset,
                                trainer=trainer_instance)

        experiment_runner = ExperimentRunner(experiment_list=[experiment])
        experiment_runner.run(path=args.path)


if __name__ == '__main__':
    main()
