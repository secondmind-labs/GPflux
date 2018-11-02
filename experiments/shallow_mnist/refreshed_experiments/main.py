from keras.datasets import mnist, fashion_mnist

from experiments.shallow_mnist.refreshed_experiments.conv_gp.creators import convgp_creator
from experiments.shallow_mnist.refreshed_experiments.nn.creators import mnist_cnn_creator, \
    cifar_cnn_creator
from experiments.shallow_mnist.refreshed_experiments.nn.configs import MNISTCNNConfiguration, \
    CifarCNNConfiguration
from experiments.shallow_mnist.refreshed_experiments.conv_gp.configs import ConvGPConfig
from experiments.shallow_mnist.refreshed_experiments.experiment_infrastructure import Experiment, \
    ExperimentSuite, KerasNNTrainer, GPTrainer
from experiments.shallow_mnist.refreshed_experiments.datasets import grey_cifar10
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
- ability to use autocompletion when implementing stuff - this can be achieved by delating with
python objects

The following setup is merely an example.
"""


def main():
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

    experiment_suite = ExperimentSuite(experiment_list=[cnn_experiment_mnist,
                                                        cnn_experiment_fashion_mnist,
                                                        cnn_experiment_cifar10])
    experiment_suite.run()


if __name__ == '__main__':
    main()
