from refreshed_experiments.configs import ConvGPConfig, MNISTCNNConfiguration, CifarCNNConfiguration
from refreshed_experiments.creators import convgp_creator, mnist_cnn_creator, cifar_cnn_creator
from refreshed_experiments.data_infrastructure import ImageClassificationDataset, \
    MaxNormalisingPreprocessor
from refreshed_experiments.datasets import mnist, mnist_5percent, mnist_10percent, mnist_25percent, \
    grey_cifar10, grey_cifar10_5percent, grey_cifar10_10percent, grey_cifar10_25percent, \
    fashion_mnist, fashion_mnist_5percent, fashion_mnist_10percent, fashion_mnist_25percent, \
    mnist_10epc, mnist_1epc, mnist_500epc, mnist_100epc, fashion_mnist_1epc, fashion_mnist_10epc, \
    fashion_mnist_100epc, fashion_mnist_500epc
from refreshed_experiments.experiment_infrastructure import Experiment
from refreshed_experiments.trainers import KerasNNTrainer, ClassificationGPTrainer

"""
python main.py -d mnist -m cnn_mnist -t classification_trainer -c small_config
"""


_convgp_mnist_exp = \
    [
        Experiment('convgp_mnist',
                   trainer=ClassificationGPTrainer(convgp_creator, config=ConvGPConfig()),
                   dataset=ImageClassificationDataset.from_keras_format(mnist),
                   dataset_preprocessor=MaxNormalisingPreprocessor),
        Experiment('convgp_mnist5percent',
                   trainer=ClassificationGPTrainer(convgp_creator, config=ConvGPConfig()),
                   dataset=ImageClassificationDataset.from_keras_format(mnist_5percent),
                   dataset_preprocessor=MaxNormalisingPreprocessor),
        Experiment('convgp_mnist10percent',
                   trainer=ClassificationGPTrainer(convgp_creator, config=ConvGPConfig()),
                   dataset=ImageClassificationDataset.from_keras_format(mnist_10percent),
                   dataset_preprocessor=MaxNormalisingPreprocessor),
        Experiment('convgp_mnist25percent',
                   trainer=ClassificationGPTrainer(convgp_creator, config=ConvGPConfig()),
                   dataset=ImageClassificationDataset.from_keras_format(mnist_25percent),
                   dataset_preprocessor=MaxNormalisingPreprocessor),
        Experiment('convgp_grey_cifar10_5percent',
                   trainer=ClassificationGPTrainer(convgp_creator, config=ConvGPConfig()),
                   dataset=ImageClassificationDataset.from_keras_format(grey_cifar10_5percent),
                   dataset_preprocessor=MaxNormalisingPreprocessor),
        Experiment('convgp_grey_cifar10_10percent',
                   trainer=ClassificationGPTrainer(convgp_creator, config=ConvGPConfig()),
                   dataset=ImageClassificationDataset.from_keras_format(grey_cifar10_10percent),
                   dataset_preprocessor=MaxNormalisingPreprocessor),
        Experiment('convgp_grey_cifar10_25percent',
                   trainer=ClassificationGPTrainer(convgp_creator, config=ConvGPConfig()),
                   dataset=ImageClassificationDataset.from_keras_format(grey_cifar10_25percent),
                   dataset_preprocessor=MaxNormalisingPreprocessor),
        Experiment('convgp_grey_cifar10',
                   trainer=ClassificationGPTrainer(convgp_creator, config=ConvGPConfig()),
                   dataset=ImageClassificationDataset.from_keras_format(grey_cifar10),
                   dataset_preprocessor=MaxNormalisingPreprocessor),
    ]


def _create_mnist_exp():
    exps = []
    name_dataset_list = [
        ('cnn_mnist', mnist),
        ('cnn_mnist_5percent', mnist_5percent),
        ('cnn_mnist_10percent', mnist_10percent),
        ('cnn_mnist_25percent', mnist_25percent),
        ('cnn_mnist_1epc', mnist_1epc),
        ('cnn_mnist_10epc', mnist_10epc),
        ('cnn_mnist_100epc', mnist_100epc),
        ('cnn_mnist_500epc', mnist_500epc),

    ]
    for early_stopping in [True, False]:
        for (name, dataset) in name_dataset_list:
            exps.append(
                Experiment(name + ('_early_stopping' if early_stopping else ''),
                           trainer=KerasNNTrainer(mnist_cnn_creator,
                                                  config=MNISTCNNConfiguration(
                                                      early_stopping=early_stopping)),
                           dataset=ImageClassificationDataset.from_keras_format(
                               dataset),
                           dataset_preprocessor=MaxNormalisingPreprocessor),
            )
    return exps


def _create_fashion_mnist_exp():
    exps = []
    name_dataset_list = [
        ('cnn_fashion_mnist', fashion_mnist),
        ('cnn_fashion_mnist_5percent', fashion_mnist_5percent),
        ('cnn_fashion_mnist_10percent', fashion_mnist_10percent),
        ('cnn_fashion_mnist_25percent', fashion_mnist_25percent),
        ('cnn_fashion_mnist_1epc', fashion_mnist_1epc),
        ('cnn_fashion_mnist_10epc', fashion_mnist_10epc),
        ('cnn_fashion_mnist_100epc', fashion_mnist_100epc),
        ('cnn_fashion_mnist_500epc', fashion_mnist_500epc),

    ]
    for early_stopping in [True, False]:
        for (name, dataset) in name_dataset_list:
            exps.append(
                Experiment(name + ('_early_stopping' if early_stopping else ''),
                           trainer=KerasNNTrainer(mnist_cnn_creator,
                                                  config=MNISTCNNConfiguration(
                                                      early_stopping=early_stopping)),
                           dataset=ImageClassificationDataset.from_keras_format(
                               dataset),
                           dataset_preprocessor=MaxNormalisingPreprocessor),
            )
    return exps


def _create_cifar_exp():
    exps = []
    name_dataset_list = [
        ('cnn_grey_cifar10', grey_cifar10),
        ('cnn_grey_cifar10_5percent', grey_cifar10_5percent),
        ('cnn_grey_cifar10_10percent', grey_cifar10_10percent),
        ('cnn_grey_cifar10_25percent', grey_cifar10_25percent),

    ]
    for early_stopping in [True, False]:
        for (name, dataset) in name_dataset_list:
            exps.append(
                Experiment(name + ('_early_stopping' if early_stopping else ''),
                           trainer=KerasNNTrainer(cifar_cnn_creator,
                                                  config=CifarCNNConfiguration(
                                                      early_stopping=early_stopping)),
                           dataset=ImageClassificationDataset.from_keras_format(
                               grey_cifar10_25percent),
                           dataset_preprocessor=MaxNormalisingPreprocessor),
            )
    return exps


_mnist_experiments = _create_mnist_exp()
_fashion_mnist_experiments = _create_fashion_mnist_exp()
_cifar_experiments = _create_cifar_exp()
_experiments = _mnist_experiments + _fashion_mnist_experiments + _cifar_experiments + \
               _convgp_mnist_exp


def get_experiment_dict():
    return {e.name: e for e in _experiments}
