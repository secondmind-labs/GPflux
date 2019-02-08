# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import pickle
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, List, cast

import numpy as np

ScalarSequence = NamedTuple('ScalarSequence', [('values', np.ndarray),
                                               ('name', str),
                                               ('x_axis_name', str),
                                               ('y_axis_name', str)])


class Scalar:
    def __init__(self, value, name, rounding=3):
        self.value = format(float(value), '.{}f'.format(rounding))
        self.name = name


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
