from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
COLORS = ['red', 'blue', 'green', 'purple', 'grey', 'yellow', 'magenta']

def plot_stat(axis, name, i):

    plt.plot(axis, d['test_'+name], COLORS[i], linestyle='-')
    plt.plot(axis, d['train_'+name], COLORS[i], linestyle='--')


directory = 'temp/test/random_mnist_1epc_max_normalised'
curr_dir = Path(directory)
for i, dir in enumerate(curr_dir.iterdir()):
    if dir.is_file():
        continue
    d = pickle.load(open('{}/training_summary.c'.format(dir), 'rb'))
    print(dir.stem, COLORS[i])
    gp_units = range(1, len(d['test_avg_nll_list']) * 100 + 1, 100)
    nn_units = range(1, len(d['test_avg_nll_list']) * 10 + 1, 10)
    x_axis = gp_units if not 'CNN' in str(dir) else nn_units

    # plt.plot(x_axis, (d['test_avg_nll_list']), COLORS[i], linestyle='-')
    # if not 'CNN' in str(dir):
    #     continue
    # plt.plot(x_axis, (d['train_avg_nll_list']), COLORS[i], linestyle='--')

    # plot_stat(x_axis, 'error_rate_list', i)
    # plot_stat(x_axis, 'avg_nll_list', i)
    # plt.plot(x_axis, (d['lenghtscales']), COLORS[i], linestyle='-')
    # plt.plot(x_axis, (d['variances']), COLORS[i], linestyle='-')
    # plt.plot(x_axis, [x.reshape(-1) for x in d['test_variance_f']], COLORS[i], linestyle='-')
    #plt.plot(x_axis, [x.reshape(-1) for x in d['train_variance_f']], COLORS[i], linestyle='-')
    if 'CNN' in str(dir):
        traj = np.concatenate([x[:100,None] for x in d['test_predictions']], axis=1).T
    else:
        traj = np.stack(d['test_predictions'])[:,:,0].T
    for row in traj:
        print(x_axis, row.shape)
        plt.plot(x_axis, row, COLORS[i])
    plt.xscale('log')

plt.show()
