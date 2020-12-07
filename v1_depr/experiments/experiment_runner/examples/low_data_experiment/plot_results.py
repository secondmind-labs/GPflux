import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn

COLORS = ['red', 'blue', 'green', 'purple', 'grey', 'yellow', 'magenta']

NAME_DICT = {'BasicCNN_KerasConfig_StatsGatheringKerasClassificationLearner': 'CNN',
             'ShallowConvGP_TickConvGPNoWeightsConfig_StatsGatheringGPClassificator': 'TICK No Weights',
             'ShallowConvGP_TickConvGPConfig_StatsGatheringGPClassificator': 'TICK'}

NAME_TO_COLOR = {'CNN': 'red', 'TICK': 'green', 'TICK No Weights': 'blue'}


def folder_to_name(dirname):
    return '_'.join(dirname.split('_')[:-1])


def plot_stat(axis, name, i, model_name):
    SUMSAMPLE = 5
    legend_name = NAME_DICT[model_name]
    plt.plot(axis[::SUMSAMPLE], d['test_' + name][::SUMSAMPLE], NAME_TO_COLOR[legend_name],
             linestyle='-', linewidth=0.8, marker='x', markersize=1,
             label='{} test_'.format(legend_name) + name)
    plt.plot(axis, d['train_' + name], NAME_TO_COLOR[legend_name], linestyle='--', marker='x',
             markersize=1,
             label='{} train_'.format(legend_name) + name, linewidth=0.8)


def sort_every_row(m):
    rows = []
    for row in m:
        rows.append(list(sorted(row)))
    return np.stack(rows)


for folder in ['random_mnist_1epc_max_normalised', 'random_mnist_10epc_max_normalised',
               'random_mnist_100epc_max_normalised']:
    directory = 'examples/saved_results/test/' + folder
    curr_dir = Path(directory)
    for i, dir in enumerate(curr_dir.iterdir()):
        if dir.is_file():
            continue
        d = pickle.load(open('{}/training_summary.c'.format(dir), 'rb'))
        gp_units = range(1, len(d['test_avg_nll_list']) * 100 + 1, 100)
        nn_units = range(1, len(d['test_avg_nll_list']) * 10 + 1, 10)
        x_axis = gp_units if not 'CNN' in str(dir) else nn_units

        try:
            pass
            # plot_stat(x_axis, 'error_rate_list', i, folder_to_name(dir.stem))
            # plt.legend(prop={'size': 6})
            # plt.xscale('log')
            # # plt.yscale('log')
            # plt.xlabel('optimisation steps')
            # plt.ylabel('error rate')
            # plt.title(dir.parent.stem)

            # plot_stat(x_axis, 'avg_nll_list', i, folder_to_name(dir.stem))
            # plt.legend(prop={'size': 6})
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.xlabel('optimisation steps')
            # plt.ylabel('negative log likelihood')
            # plt.title(dir.parent.stem)

            # plt.plot(x_axis, d['lenghtscales'], COLORS[i], linestyle='-', label=folder_to_name(dir.stem))
            # plt.legend(prop={'size': 6})
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.xlabel('optimisation steps')
            # plt.ylabel('lenghtscales')

            plt.plot(x_axis, (d['variances']), NAME_TO_COLOR[NAME_DICT[folder_to_name(dir.stem)]],
                     linestyle='-', label=folder_to_name(dir.stem))
            plt.legend(prop={'size': 6})
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('optimisation steps')
            plt.ylabel('variance')

            # plt.plot(x_axis, [x.reshape(-1) for x in d['test_variance_f']], COLORS[i],
            #          linestyle='-', linewidth=0.01, marker='x', markersize=1)
            # plt.legend(prop={'size': 6})
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.xlabel('optimisation steps')
            # plt.ylabel('test variance')

            # plt.plot(x_axis[::10], [x.reshape(-1) for x in d['train_variance_f']][::10], COLORS[i],
            #          linestyle='-', linewidth=0.1, marker='x', markersize=2)
            # plt.legend(prop={'size': 6})
            # plt.xscale('log')
            # plt.yscale('log')
            # plt.xlabel('optimisation steps')
            # plt.ylabel('train variance')

            # if 'CNN' not in str(dir.stem):
            #     continue

            # if 'CNN' in str(dir):
            #     traj = np.stack(d['test_predictions'])
            # else:
            #     traj = np.stack(d['test_predictions'])[:, :, 0]
            #
            # # traj = sort_every_row(traj)
            # print(traj.shape)
            # traj = traj[:,:2000]
            # plt.plot(x_axis, traj, NAME_TO_COLOR[NAME_DICT[folder_to_name(dir.stem)]],
            #              linewidth=0.0, marker='.', markersize=0.01)
            # plt.xlabel('optimisation steps')
            # plt.ylabel('correct test label probability')
            # plt.yscale('log')

        except KeyError:
            pass

    plt.savefig('var' + dir.parent.stem + dir.stem)
    plt.close()
