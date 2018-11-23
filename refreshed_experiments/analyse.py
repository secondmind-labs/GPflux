import os
import pickle
import argparse


def read_results(results_path):
    result_folders = os.listdir(results_path)
    results_dict = {}
    summary_dict = {}
    for folder in sorted(result_folders):
        training_summary_path = os.path.join(results_path, folder, 'training_summary.c')
        results_dict[folder] = pickle.load(open(training_summary_path, 'rb'))
        summary_path = os.path.join(results_path, folder, 'summary.txt')
        summary_dict[folder] = str('\n'.join(open(summary_path, 'r').readlines()))
    return results_dict, summary_dict


def _plot_train_valid(plot_path, results_dict, stats):
    import matplotlib.pyplot as plt
    for stat in stats:
        training_stat = results_dict['val' + stat]
        validation_stat = results_dict[stat]
        plt.plot(training_stat)
        plt.plot(validation_stat)
    plt.show()


def generate_report():
    latex_template = """"""
    return latex_template


def gather_to_table(results_dict, stats):
    tables = {stat: [] for stat in stats}
    for stat in sorted(stats):
        for name in sorted(results_dict.keys()):
            r = results_dict[name]['final_' + stat]
            tables[stat].append(r)


def get_report_str(results_dict, summary_dict, name, stats):
    summary_str = '-----------------------------------------------------------------------------\n'
    summary_str += 'Name: ' + name + '\n\n'
    summary_str += summary_dict[name] + '\n'
    for stat in sorted(stats):
        summary_str += ('final test {} {}'.format(
            results_dict[name]['final_test_' + stat],
            stat)) + '\n'
    summary_str += '-----------------------------------------------------------------------------\n'
    return summary_str


def analyse(results_path, stat):
    results_dict, summart_dict = read_results(results_path)
    for name in results_dict.keys():
        report_str = get_report_str(results_dict, summart_dict, name, stat)
        print(report_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrypoint for plotting the results:\n {}')
    parser.add_argument('--results_path', '-r', help='The path to the result.',
                        type=str, required=True)
    parser.add_argument('--stats', '-s', help='The statistic to plot', type=str,
                        required=True, nargs='+')

    args = parser.parse_args()
    analyse(args.results_path, args.stats)
