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


def _get_dataset(dataset_name):
    from experiments.shallow_mnist.refreshed_experiments.datasets \
        import \
        mnist, \
        grey_cifar10, \
        grey_cifar100, \
        mnist_10percent, \
        mnist_25percent, \
        mnist_50percent
    datasets = mnist, \
               grey_cifar10, \
               grey_cifar100, \
               mnist_10percent, \
               mnist_25percent, \
               mnist_50percent
    dataset_dict = {d.__name__: d for d in datasets}
    try:
        return dataset_dict[dataset_name]
    except KeyError:
        raise KeyError('Dataset {} not found. Available are: {}'.format(dataset_name, ' '.join(
            dataset_dict.keys())))


def _get_model_creator(model_creator_name):
    from experiments.shallow_mnist.refreshed_experiments.nn.creators \
        import \
        cifar_cnn_creator, \
        mnist_cnn_creator, \
        mnist_fashion_cnn_creator
    model_creators = cifar_cnn_creator, \
                     mnist_cnn_creator, \
                     mnist_fashion_cnn_creator
    model_creator_dict = {mc.__name__: mc for mc in model_creators}
    try:
        return model_creator_dict[model_creator_name]
    except KeyError:
        raise KeyError('Model creator {} '
                       'not found. Available are: {}'.format(model_creator_name,
                                                             ' '.join(
                                                                 model_creator_dict.keys())))


def _get_model_config(config_name):
    from experiments.shallow_mnist.refreshed_experiments.nn.configs \
        import \
        MNISTCNNConfiguration, \
        CifarCNNConfiguration
    configs = MNISTCNNConfiguration, \
              CifarCNNConfiguration
    configs_dict = {c.__name__: c for c in configs}
    try:
        return configs_dict[config_name]
    except KeyError:
        raise KeyError('Config {} not found. Available are: {}'.format(config_name,
                                                                       ' '.join(
                                                                           configs_dict.keys())))


def main():
    parser = argparse.ArgumentParser(description='Entrypoint for running the experiments.')
    parser.add_argument('--model_creator', help='the model creator', type=str)
    parser.add_argument('--model_configuration', help='the model configuration', type=str)
    parser.add_argument('--dataset', help='dataset', type=str)
    parser.add_argument('--trainer', help='used_trainer', type=str)

    args = parser.parse_args()
    dataset = _get_dataset(args.dataset)
    model_creator = _get_model_creator(args.model_creator)
    config = _get_model_config(args.config)

    gp_experiment_mnist = Experiment('convgp_experiment',
                                     trainer=GPTrainer(convgp_creator, config=ConvGPConfig),
                                     dataset=ImageClassificationDataset.from_keras_format(
                                         mnist),
                                     dataset_preprocessor=DummyPreprocessor)

    cnn_experiment_mnist = Experiment('cnn_experiment',
                                      trainer=KerasNNTrainer(mnist_cnn_creator,
                                                             config=MNISTCNNConfiguration),
                                      dataset=ImageClassificationDataset.from_keras_format(
                                          mnist),
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


if __name__ == '__main__':
    main()
