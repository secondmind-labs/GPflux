# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import multiprocessing
import os
import multiprocessing as mp
import time
from collections import namedtuple

from experiments.experiment_runner.run import parametrised_main

ExperimentSpecification = namedtuple('ExperimentSpecification',
                                     'name, creator, dataset, config, learner')


def run_experiment(experiment: ExperimentSpecification, available_gpus, path):
    while available_gpus.empty():
        time.sleep(0.01)
    gpu_num = available_gpus.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    parametrised_main(experiment.config, experiment.creator, experiment.learner,
                      experiment.dataset.load_data(),
                      1, path, experiment.name)
    available_gpus.put(gpu_num)


def run_multiple(experiment_list, gpus, path):
    manager = multiprocessing.Manager()
    available_gpus = manager.Queue(maxsize=len(gpus))
    for gpu_id in gpus:
        available_gpus.put(gpu_id)
    with mp.Pool(len(gpus)) as pool:
        pool.starmap(run_experiment, [(exp, available_gpus, path) for exp in experiment_list])
