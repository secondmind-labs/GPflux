import argparse
from pathlib import Path

from refreshed_experiments.experiment_infrastructure import ExperimentRunner, Experiment
from refreshed_experiments.utils import get_from_module
from refreshed_experiments import configs, model_creators, trainers, datasets

"""
Entrypoint for running experiments.
"""


def get_name(trainer, config, creator, dataset):
    return "experiment-{}-{}-{}-{}".format(trainer.name, config.name, creator.__name__,
                                dataset.name)


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
    parser.add_argument('--repetitions', '-r', help='The number of repetitions of the experiment',
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
