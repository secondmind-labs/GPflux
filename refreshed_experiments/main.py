import argparse
from pathlib import Path

from refreshed_experiments.experiments import get_experiment_dict
from refreshed_experiments.experiment_infrastructure import ExperimentRunner, Experiment
from refreshed_experiments.utils import get_from_module
from refreshed_experiments import configs, creators, trainers, datasets

"""
Entrypoint for running experiments.
"""

def get_name(trainer, config, creator, dataset):
    return "{}_{}_{}_{}".format(trainer.name, config.name, creator.name, dataset.name)

def main():
    parser = argparse.ArgumentParser(
        description="""Entrypoint for running the experiments. Run with:
        python main.py -d dataset -mc model_creator -t trainer -c config
        Available are:\n""".format())

    parser.add_argument('--model_creator', '-mc', help='The names of the experiments to run.',
                        type=str, required=True)
    parser.add_argument('--config', '-c', help='config.',
                        type=str, required=True)
    parser.add_argument('--dataset', '-d', help='dataset.',
                        type=str, required=True)
    parser.add_argument('--trainer', '-t', help='trainer.',
                        type=str, required=True)
    parser.add_argument('--path', '-p', help='The path were results will be stored', type=Path,
                        required=True)

    args = parser.parse_args()

    config = get_from_module(args.config, configs)
    model_creator = get_from_module(args.model_creator, creators)
    trainer = get_from_module(args.trainer, trainers)
    dataset = get_from_module(args.dataset, datasets)

    trainer_instance = trainer(model_creator=model_creator,
                               config=config)

    experiment = Experiment(name=get_name(trainer_instance, config, model_creator, dataset),
                            dataset=dataset,
                            trainer=trainer_instance)

    experiment_runner = ExperimentRunner(experiment_list=[experiment])
    experiment_runner.run(path=args.path)


def _main():
    experiments_dict = get_experiment_dict()

    parser = argparse.ArgumentParser(
        description='Entrypoint for running the experiments. Available are:\n {}'.format(
            ' '.join(experiments_dict.keys())))
    parser.add_argument('--experiment_names', '-e', help='The names of the experiments to run.',
                        type=str, nargs='+', required=True)
    parser.add_argument('--path', '-p', help='The path were results will be stored', type=Path,
                        required=True)

    args = parser.parse_args()

    experiments = []
    for name in args.experiment_names:
        try:
            experiments.append(experiments_dict[name])

        except KeyError:
            raise KeyError('Experiment {} not found. '
                           'Available are: {}'.format(name, ' '.join(experiments_dict.keys())))

    experiment_suite = ExperimentRunner(experiment_list=experiments)
    experiment_suite.run(path=args.path)


if __name__ == '__main__':
    main()
