import argparse
from keras.datasets import mnist, fashion_mnist

from experiments.shallow_mnist.refreshed_experiments.conv_gp.creators import convgp_creator
from experiments.shallow_mnist.refreshed_experiments.datasets import grey_cifar10
from experiments.shallow_mnist.refreshed_experiments.nn.creators import mnist_cnn_creator, \
    cifar_cnn_creator
from experiments.shallow_mnist.refreshed_experiments.nn.configs import MNISTCNNConfiguration, \
    CifarCNNConfiguration
from experiments.shallow_mnist.refreshed_experiments.conv_gp.configs import ConvGPConfig
from experiments.shallow_mnist.refreshed_experiments.experiment_infrastructure import Experiment, \
    ExperimentSuite, KerasNNTrainer, GPTrainer
from experiments.shallow_mnist.refreshed_experiments.data_infrastructure import DummyPreprocessor, \
    ImageClassificationDataset

"""
Main entrypoint - here we need to discuss how would you like to run the experiments.
Things to consider:
- ease of use
- ability to loop over different configurations for a fixed dataset
- ability to quickly prototype
- ability to run a model vs different optimisation setups, i.e. don't change the inference scheme, 
but change the optimisation procedure
- ability to run within docker images so we can run the experiments anywhere if needed
- ability to use autocompletion when implementing stuff - this can be achieved by dealing with
python objects

The following setup is merely an example.
"""


def _get_experiment_dict():
    experiments = [
        Experiment('basic_convgp_mnist_exp',
                   trainer=GPTrainer(convgp_creator, config=ConvGPConfig),
                   dataset=ImageClassificationDataset.from_keras_format(mnist),
                   dataset_preprocessor=DummyPreprocessor),
        Experiment('cnn_experiment_mnist',
                   trainer=KerasNNTrainer(mnist_cnn_creator,
                                          config=MNISTCNNConfiguration),
                   dataset=ImageClassificationDataset.from_keras_format(
                       mnist),
                   dataset_preprocessor=DummyPreprocessor),
        Experiment('cnn_experiment_fashion_mnist',
                   trainer=KerasNNTrainer(mnist_cnn_creator,
                                          config=MNISTCNNConfiguration),
                   dataset=ImageClassificationDataset.from_keras_format(
                       fashion_mnist),
                   dataset_preprocessor=DummyPreprocessor),
        Experiment('cnn_experiment_cifar10',
                   trainer=KerasNNTrainer(cifar_cnn_creator,
                                          config=CifarCNNConfiguration),
                   dataset=ImageClassificationDataset.from_keras_format(
                       grey_cifar10),
                   dataset_preprocessor=DummyPreprocessor),
    ]
    return {e.name: e for e in experiments}


def main():
    parser = argparse.ArgumentParser(description='Entrypoint for running the experiments.')
    parser.add_argument('--experiment_names', '-e', help='The names of the experiments to run.',
                        type=str, nargs='+', required=True)

    args = parser.parse_args()

    experiments_dict = _get_experiment_dict()
    experiments = []
    for name in args.experiment_names:
        try:
            experiments.append(experiments_dict[name])

        except KeyError:
            raise KeyError('Experiment {} not found. '
                           'Available are: {}'.format(name, ' '.join(experiments_dict.keys())))

    experiment_suite = ExperimentSuite(experiment_list=experiments)
    experiment_suite.run()


if __name__ == '__main__':
    main()
