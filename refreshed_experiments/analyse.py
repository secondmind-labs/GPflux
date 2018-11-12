import os
import pickle
import argparse


def read_results(results_path):
    result_folders = os.listdir(results_path)
    results_dict = {}
    summary_dict = {}
    for folder in result_folders:
        results_dict[folder] = pickle.load(open('results/' + folder + '/training_summary.c', 'rb'))
        summary_dict[folder] = str(
            '\n'.join(open('results/' + folder + '/summary.txt', 'r').readlines()))
    return results_dict, summary_dict


def plot(results_dict, summary_dict, name, stat):
    import matplotlib.pyplot as plt
    plt.plot(results_dict[name][stat])
    plt.plot(results_dict[name]['val_' + stat])
    print(summary_dict[name])
    print('train {} {} test {} {}'.format(results_dict[name][stat][-1], stat,
                                          results_dict[name]['val_' + stat][-1], stat))
    plt.show()


def analyse(results_path, stat):
    results_dict, summart_dict = read_results(results_path)
    for name in results_dict.keys():
        plot(results_path, summart_dict, name, stat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrypoint for plotting the results:\n {}')
    parser.add_argument('--results_path', '-r', help='The path to the result.',
                        type=str, required=True)
    parser.add_argument('--stat', '-s', help='The statistic to plot', type=str,
                        required=True)

    args = parser.parse_args()
    analyse(args.results_path, args.stat)
