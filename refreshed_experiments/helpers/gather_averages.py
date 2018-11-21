import argparse

from refreshed_experiments.analyse import analyse_average

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Entrypoint for plotting the results:\n {}')
    parser.add_argument('--results_path', '-r', help='The path to the result.',
                        type=str, required=True)
    parser.add_argument('--stats', '-s', help='The statistic to plot', type=str,
                        required=True, nargs='+')
    parser.add_argument('--name_prefix', '-np', help='The name prefix of a method to gather',
                        type=str,
                        required=True, nargs='+')

    args = parser.parse_args()
    for name in args.name_prefix:
        analyse_average(args.results_path, name, args.stats)
