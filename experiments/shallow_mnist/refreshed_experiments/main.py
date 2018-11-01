import argparse

from keras.datasets import mnist, fashion_mnist

from experiments.shallow_mnist.refreshed_experiments.conv_gp.creators import convgp_creator
from experiments.shallow_mnist.refreshed_experiments.nn.creators import mnist_cnn_creator, \
    cifar_cnn_creator
from experiments.shallow_mnist.refreshed_experiments.nn.configs import MNISTCNNConfiguration, CifarCNNConfiguration
from experiments.shallow_mnist.refreshed_experiments.conv_gp.configs import ConvGPConfig
from experiments.shallow_mnist.refreshed_experiments.experiment_infrastructure import Experiment, \
    ExperimentSuite, KerasNNTrainer, GPTrainer
from experiments.shallow_mnist.refreshed_experiments.datasets import grey_cifar10
from experiments.shallow_mnist.refreshed_experiments.data_infrastructure import DummyPreprocessor, \
    ImageClassificationDataset


def main():
    parser = argparse.ArgumentParser(description='Entrypoint for running experiments.')
    parser.add_argument('-datasets', '--datasets', nargs='+',
                        help='The experiments will be executed on these datasets.'
                             'Available: ')
    parser.add_argument('-models', '--models', nargs='+',
                        help='The experiments will be executed on these models.'
                             'Available: ')

    args = parser.parse_args()

    gp_experiment_mnist = Experiment('convgp_experiment',
                                     trainer=GPTrainer(convgp_creator, config=ConvGPConfig),
                                     dataset=ImageClassificationDataset.from_keras_format(mnist),
                                     dataset_preprocessor=DummyPreprocessor)

    cnn_experiment_mnist = Experiment('cnn_experiment',
                                      trainer=KerasNNTrainer(mnist_cnn_creator,
                                                             config=MNISTCNNConfiguration),
                                      dataset=ImageClassificationDataset.from_keras_format(mnist),
                                      dataset_preprocessor=DummyPreprocessor)

    cnn_experiment_fashion_mnist = Experiment('cnn_experiment',
                                              trainer=KerasNNTrainer(mnist_cnn_creator,
                                                                     config=MNISTCNNConfiguration),
                                              dataset=ImageClassificationDataset.from_keras_format(
                                                  fashion_mnist),
                                              dataset_preprocessor=DummyPreprocessor)

    cnn_experiment_cifar10 = Experiment('cnn_experiment',
                                        trainer=KerasNNTrainer(cifar_cnn_creator,
                                                               config=CifarCNNConfiguration),
                                        dataset=ImageClassificationDataset.from_keras_format(
                                            grey_cifar10),
                                        dataset_preprocessor=DummyPreprocessor)

    experiment_suite = ExperimentSuite(experiment_list=[cnn_experiment_mnist])
    experiment_suite.run()

    # histories = experiment.run()
    # train_resutls, test_results = group_histories(histories)
    # plt.plot(np.log(train_resutls.T), np.log(test_results).T, 'b.-', linewidth=0.8)
    # # plt.plot(range(train_resutls[0].size), range(train_resutls[0].size), 'r--')
    # plt.xlabel(r'$\log\ train\ loss$')
    # plt.ylabel(r'$\log\ test\ loss$')
    # plt.show()


if __name__ == '__main__':
    main()
