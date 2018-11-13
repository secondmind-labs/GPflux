import os
import pickle
import argparse


def read_results(results_path):
    result_folders = os.listdir(results_path)
    results_dict = {}
    summary_dict = {}
    for folder in result_folders:
        training_summary_path = os.path.join(results_path, folder, 'training_summary.c')
        results_dict[folder] = pickle.load(open(training_summary_path, 'rb'))
        summary_path = os.path.join(results_path, folder, 'summary.txt')
        summary_dict[folder] = str('\n'.join(open(summary_path, 'r').readlines()))
    return results_dict, summary_dict


def plot(results_dict, summary_dict, name, stats):
    import matplotlib.pyplot as plt
    summary_str = '--------------------------------------\n'
    summary_str += 'Experiment ' + ' '.join(name.split('_')[:-1]) + '\n'
    summary_str += 'Hyperparameters:\n'
    summary_str += summary_dict[name] + '\n'
    # plt.plot(results_dict[name][stat])
    # plt.plot(results_dict[name]['val_' + stat])
    for stat in stats:
        summary_str += ('Final train {} {}, final test {} {}'.format(
            results_dict[name]['final_' + stat], stat,
            results_dict[name]['final_val_' + stat],
            stat)) + '\n'
    summary_str += '--------------------------------------\n'
    print(summary_str)
    # plt.show()


def analyse(results_path, stat):
    results_dict, summart_dict = read_results(results_path)
    for name in results_dict.keys():
        plot(results_dict, summart_dict, name, stat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrypoint for plotting the results:\n {}')
    parser.add_argument('--results_path', '-r', help='The path to the result.',
                        type=str, required=True)
    parser.add_argument('--stats', '-s', help='The statistic to plot', type=str,
                        required=True, nargs='+')

    args = parser.parse_args()
    analyse(args.results_path, args.stats)
