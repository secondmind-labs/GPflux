from refreshed_experiments.utils import Arguments


def cnn_all_datasets(path):
    return [
        Arguments(creator='basic_cnn_creator',
                  dataset=dataset,
                  config='BasicCNNConfig',
                  trainer='KerasClassificator',
                  path=path,
                  repetitions='10') for dataset in
        ['mnist', 'fashion_mnist', 'grey_cifar10', 'svhn']
    ]


def tickconvgp_all_datasets(path):
    return [
        Arguments(creator='convgp_creator',
                  dataset=dataset,
                  config='TickConvGPConfig',
                  trainer='GPClassificator',
                  path=path,
                  repetitions='1') for dataset in
        ['mnist', 'fashion_mnist', 'grey_cifar10', 'svhn']
    ]


def convgp_all_datasets(path):
    return [
        Arguments(creator='convgp_creator',
                  dataset=dataset,
                  config='ConvGPConfig',
                  trainer='GPClassificator',
                  path=path,
                  repetitions='1') for dataset in
        ['mnist', 'fashion_mnist', 'grey_cifar10', 'svhn']
    ]


def _create_fractions(dataset):
    def f(path):
        return [
            Arguments(creator='basic_cnn_creator',
                      dataset=d,
                      config='BasicCNNConfig',
                      trainer='KerasClassificator',
                      path=path,
                      repetitions='10') for d in
            ['{}_100epc'.format(dataset),
             '{}_5percent'.format(dataset),
             '{}_10percent'.format(dataset),
             '{}_25percent'.format(dataset)]
        ]

    return f


mnist_fractions = _create_fractions('mnist')
fashion_mnist_fractions = _create_fractions('fashion_mnist')
svhn_fractions = _create_fractions('svhn')
grey_cifar10_fractions = _create_fractions('grey_cifar10')
