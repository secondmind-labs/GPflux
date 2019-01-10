# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from pathlib import Path

from refreshed_experiments.refactored.results_managing import DatasetReport


def read():
    path = 'test/my_experiment'
    report_creator = DatasetReport(Path(path))
    report = report_creator.create_txt_report()
    report_creator.plot_summaries(Path('test_plots'))
    print(report)


if __name__ == '__main__':
    read()
