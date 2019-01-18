# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import multiprocessing
import os
import subprocess
import multiprocessing as mp
import sys
import time


class ExperimentSpecification:

    def __init__(self, name, creator, dataset, config, learner):
        self.name = name
        self.creator = creator
        self.dataset = dataset
        self.config = config
        self.learner = learner
        self.file = __file__


def run_experiment(experiment: ExperimentSpecification, available_gpus, path):
    while available_gpus.empty():
        time.sleep(0.01)
    gpu_num = available_gpus.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    location = os.path.abspath(os.path.dirname(sys.modules[__name__].__file__))
    subprocess.call(['python', os.path.join(location, 'run.py'),
                     '-n', experiment.name,
                     '-m', experiment.creator.__name__,
                     '-d', experiment.dataset.__name__,
                     '-c', experiment.config.__name__,
                     '-l', experiment.learner.__name__,
                     '-p', path])
    available_gpus.put(gpu_num)


def run_multiple(experiment_list, gpus, path):
    manager = multiprocessing.Manager()
    available_gpus = manager.Queue(maxsize=len(gpus))
    for gpu_id in gpus:
        available_gpus.put(gpu_id)
    with mp.Pool(len(gpus)) as pool:
        pool.starmap(run_experiment, [(exp, available_gpus, path) for exp in experiment_list])
