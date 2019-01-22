# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import pickle
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, List, cast

import numpy as np
import matplotlib.pyplot as plt

from experiments.experiment_runner.utils import get_text_summary, plot_summaries

COLORS = ['blue', 'red', 'green', 'yellow']

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

    def plot_common_summary(self, save_path):
        """
        Creates a figure depicting train and test averages of loss and error rates. Uses log
        scale on x axis
        """
        STEPS_PER_X_UNIT = 250
        num_datasets = len(list(self._path.iterdir()))
        g, ax = plt.subplots(num_datasets, 2, sharex=True)
        for n, dataset_dir in enumerate(self._path.iterdir()):
            train_loss_averages = defaultdict(list)
            test_loss_averages = defaultdict(list)
            train_error_rate_averages = defaultdict(list)
            test_error_rate_averages = defaultdict(list)
            for experiment_dir in dataset_dir.iterdir():
                summary = ReportCreator.get_summary(experiment_dir)
                d = defaultdict(list)
                for sequence in summary.scalar_sequences:
                    name = sequence.name.replace('test', '').replace('train', '')
                    d[name].append(sequence)
                for item in d.values():
                    f, inner_ax = plt.subplots()
                    for i, sequence in enumerate(item):
                        plt.plot(sequence.values, label=sequence.name, color=COLORS[i])
                        experiment_id = experiment_dir.stem.split('_')[0]
                        if 'train' in sequence.name:
                            if 'loss' in sequence.name:
                                train_loss_averages[
                                    '{}_{}'.format(experiment_id, sequence.name)].append(
                                    sequence.values)
                            else:
                                if 'error' in sequence.name:
                                    train_error_rate_averages[
                                        '{}_{}'.format(experiment_id, sequence.name)].append(
                                        sequence.values)

                        elif 'test' in sequence.name:
                            if 'loss' in sequence.name:
                                test_loss_averages[
                                    '{}_{}'.format(experiment_id, sequence.name)].append(
                                    sequence.values)
                            elif 'error' in sequence.name:
                                test_error_rate_averages[
                                    '{}_{}'.format(experiment_id, sequence.name)].append(
                                    sequence.values)
                    name = sequence.name.replace('test', '').replace('test', '')
                    inner_ax.set_title(name)
                    inner_ax.legend()
                    inner_ax.set_xlabel(sequence.x_axis_name)
                    inner_ax.set_ylabel(sequence.y_axis_name)
                    (save_path / Path(dataset_dir.stem) / Path(experiment_dir.stem)).mkdir(
                        exist_ok=True, parents=True)
                    f.savefig(str(save_path / Path(dataset_dir.stem) / Path(
                        experiment_dir.stem)) + '/{}.png'.format(name))
                    plt.close(f)
            stats2plot = [test_loss_averages, test_error_rate_averages]
            for i, d in enumerate(stats2plot):
                for name, values in d.items():
                    if 'CNN' in name:
                        color = '#1f77b4'
                    else:
                        color = '#ff7f0e'
                    (save_path / Path(dataset_dir.stem) / Path('averages')).mkdir(exist_ok=True,
                                                                                  parents=True)
                    y_values = np.array(values).mean(0)
                    x_range = range(1,  len(y_values)*STEPS_PER_X_UNIT, STEPS_PER_X_UNIT) \
                        if 'CNN' not in name else range(1,  len(y_values)*10, 10)
                    y_std = np.array(values).std(0)
                    ax[n][i].errorbar(x_range, y_values, yerr=y_std, c=color,
                                      label={'ShallowConvGP':'TICK-GP', 'BasicCNN':'CNN'}[name.split('_')[0]], capthick=1)
                    if 'loss' in name:
                        ax[n][i].axhline(-np.log(0.1), linestyle=':', color='darkgrey', linewidth=1.,
                                         xmin=0.04, xmax=0.96)
                    ax[n][i].set_xscale('log')
                if n < num_datasets-1:
                    ax[n][i].set_xticks([])
                else:
                    ax[n][i].set_xlabel('$optimisation\ steps$')
                ax[0][i].set_title(name.split('_')[1])
            ax[-1][-1].legend(prop={'size':6})
            g.tight_layout()
            plt.close(g)
        (save_path / Path('averages')).mkdir(exist_ok=True, parents=True)
        g.savefig(str(save_path / Path('averages')) + '/common.pdf')
