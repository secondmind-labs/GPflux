# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import argparse
from collections import defaultdict

import numpy as np

from refreshed_experiments.analyse import read_results

"""
Example usage:
```
python gather_averages.py -r /tmp/results -s loss error -np experiment-1
```

This will gather all the results stored in folders with name prefixes experiment-1 and report
average and standard deviation of loss and error. 
"""


def analyse_average(results_path, name_prefix, stats):
    results_dict, summart_dict = read_results(results_path)
    gathered_results = defaultdict(lambda: defaultdict(list))
    for stat in sorted(stats):
        for name in results_dict.keys():
            if name.startswith(name_prefix):
                gathered_results[name_prefix][stat].append(results_dict[name]['final_test_' + stat])
    for name, results in gathered_results.items():
        for stat in sorted(stats):
            print(name, stat,
                  '{0:.4f} {1:.4f}'.format(np.mean(results[stat]), np.std(results[stat])))


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
