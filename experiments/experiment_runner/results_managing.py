# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import pickle
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, List, cast

import numpy as np

from experiments.experiment_runner.utils import get_text_summary, plot_summaries

ScalarSequence = NamedTuple('ScalarSequence', [('values', np.ndarray),
                                               ('name', str),
                                               ('x_axis_name', str),
                                               ('y_axis_name', str)])
Scalar = NamedTuple('Scalar',
                    [('value', float),
                     ('name', str)])
Summary = NamedTuple('Summary',
                     [('scalars', List[Scalar]),
                      ('scalar_sequences', List[ScalarSequence])])


class ReportCreator:
    def __init__(self, path: Path):
        self._path = path

    @staticmethod
    def get_summary(path: Path):
        summary_path = path / 'training_summary.c'
        summary = pickle.load(open(str(summary_path), 'rb'))
        summary = cast(Summary, summary)
        return summary

    def create_report(self):
        scalars, scalar_sequences = self.get_summary(self._path)
        scalar_dict = defaultdict(list)
        scalar_list_dict = defaultdict(list)
        for scalar in scalars:
            scalar_dict[scalar.name].append(scalar.value)
        for scalar_sequence in scalar_sequences:
            scalar_list_dict[scalar_sequence.name] = scalar_sequence.values
        return scalar_dict, scalar_list_dict


class DatasetReport:
    def __init__(self, path: Path):
        self._path = path

    def create_txt_report(self):
        separator_length = 50
        report_str = ''
        for dataset_dir in self._path.iterdir():
            report_str += '\n' + '_' * separator_length
            report_str += '\n' + 'Dataset: ' + dataset_dir.stem
            report_str += '\n' + '-' * separator_length
            for experiment_dir in dataset_dir.iterdir():
                report_str += '\n' + experiment_dir.stem + '\n'
                summary = ReportCreator.get_summary(experiment_dir)
                report_str += get_text_summary(summary)
                report_str += '-' * separator_length
            report_str += '\n' + '_' * separator_length
        return report_str

    def plot_summaries(self, path):
        for dataset_dir in self._path.iterdir():
            for experiment_dir in dataset_dir.iterdir():
                summary = ReportCreator.get_summary(experiment_dir)
                plot_summaries(summary, path, experiment_dir.stem)

