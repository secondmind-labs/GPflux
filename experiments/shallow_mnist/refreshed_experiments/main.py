import argparse
from pathlib import Path

from experiments.shallow_mnist.refreshed_experiments.conv_gp.creators import convgp_creator
from experiments.shallow_mnist.refreshed_experiments.datasets import grey_cifar10, mnist, \
    mnist_5percent, mnist_25percent, mnist_10percent
from experiments.shallow_mnist.refreshed_experiments.nn.creators import mnist_cnn_creator, \
    cifar_cnn_creator
from experiments.shallow_mnist.refreshed_experiments.nn.configs import MNISTCNNConfiguration, \
    CifarCNNConfiguration
from experiments.shallow_mnist.refreshed_experiments.conv_gp.configs import ConvGPConfig
from experiments.shallow_mnist.refreshed_experiments.experiment_infrastructure import Experiment, \
    ExperimentSuite, KerasNNTrainer, ClassificationGPTrainer
from experiments.shallow_mnist.refreshed_experiments.data_infrastructure import DummyPreprocessor, \
    ImageClassificationDataset, MaxNormalisingPreprocessor

"""
Entrypoint for running experiments.
"""


def _get_experiment_dict():
    experiments = \
        [
            Experiment('convgp_mnist',
                       trainer=ClassificationGPTrainer(convgp_creator, config=ConvGPConfig),
                       dataset=ImageClassificationDataset.from_keras_format(mnist),
                       dataset_preprocessor=MaxNormalisingPreprocessor),
            Experiment('convgp_mnist5percent',
                       trainer=ClassificationGPTrainer(convgp_creator, config=ConvGPConfig),
                       dataset=ImageClassificationDataset.from_keras_format(mnist_5percent),
                       dataset_preprocessor=MaxNormalisingPreprocessor),
            Experiment('convgp_mnist10percent',
                       trainer=ClassificationGPTrainer(convgp_creator, config=ConvGPConfig),
                       dataset=ImageClassificationDataset.from_keras_format(mnist_10percent),
                       dataset_preprocessor=MaxNormalisingPreprocessor),
            Experiment('convgp_mnist25percent',
                       trainer=ClassificationGPTrainer(convgp_creator, config=ConvGPConfig),
                       dataset=ImageClassificationDataset.from_keras_format(mnist_25percent),
                       dataset_preprocessor=MaxNormalisingPreprocessor),
            Experiment('cnn_mnist',
                       trainer=KerasNNTrainer(mnist_cnn_creator,
                                              config=MNISTCNNConfiguration),
                       dataset=ImageClassificationDataset.from_keras_format(
                           mnist),
                       dataset_preprocessor=MaxNormalisingPreprocessor),
            Experiment('cnn_mnist_5percent',
                       trainer=KerasNNTrainer(mnist_cnn_creator,
                                              config=MNISTCNNConfiguration),
                       dataset=ImageClassificationDataset.from_keras_format(
                           mnist_5percent),
                       dataset_preprocessor=MaxNormalisingPreprocessor),
            Experiment('cnn_mnist_10percent',
                       trainer=KerasNNTrainer(mnist_cnn_creator,
                                              config=MNISTCNNConfiguration),
                       dataset=ImageClassificationDataset.from_keras_format(
                           mnist_10percent),
                       dataset_preprocessor=MaxNormalisingPreprocessor),
            Experiment('cnn_mnist_25percent',
                       trainer=KerasNNTrainer(mnist_cnn_creator,
                                              config=MNISTCNNConfiguration),
                       dataset=ImageClassificationDataset.from_keras_format(
                           mnist_25percent),
                       dataset_preprocessor=MaxNormalisingPreprocessor),
            Experiment('cnn_cifar10',
                       trainer=KerasNNTrainer(cifar_cnn_creator,
                                              config=CifarCNNConfiguration),
                       dataset=ImageClassificationDataset.from_keras_format(
                           grey_cifar10),
                       dataset_preprocessor=MaxNormalisingPreprocessor),
        ]
    return {e.name: e for e in experiments}


def main():
    experiments_dict = _get_experiment_dict()

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

    experiment_suite = ExperimentSuite(experiment_list=experiments)
    experiment_suite.run(path=args.path)


if __name__ == '__main__':
    main()
