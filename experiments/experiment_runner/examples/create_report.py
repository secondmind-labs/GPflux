# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

"""
Reads the results from the PATH_TO_RESULTS and prints a summary of them.
"""


from pathlib import Path

from experiments.experiment_runner.results_managing import DatasetReport

PATH_TO_RESULTS = 'test/my_experiment'


def read(path):
    report_creator = DatasetReport(Path(path))
    report = report_creator.create_txt_report()
    report_creator.plot_summaries(Path('test_plots'))
    print(report)


if __name__ == '__main__':
    read(PATH_TO_RESULTS)
