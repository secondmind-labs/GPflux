import multiprocessing
import os
import subprocess
import multiprocessing as mp
import time
from collections import namedtuple

PATH = '/tmp'

Arguments = namedtuple('Arguments',
                       'creator dataset config trainer path')


def run_experiment(arguments, available_gpus):
    while available_gpus.empty():
        time.sleep(0.01)
    gpu_num = available_gpus.get()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    subprocess.call(['python', 'run.py',
                     '-mc', arguments.creator,
                     '-d', arguments.dataset,
                     '-c', arguments.config,
                     '-t', arguments.trainer,
                     '-p', arguments.path])
    available_gpus.put(gpu_num)


def run_multiple(experiment_list, num_gpus):
    manager = multiprocessing.Manager()
    available_gpus = manager.Queue(maxsize=num_gpus)
    for gpu_id in range(num_gpus):
        available_gpus.put(gpu_id)
    with mp.Pool(num_gpus) as pool:
        pool.starmap(run_experiment, [(exp, available_gpus) for exp in experiment_list])


experiment_list = [
    Arguments('basic_cnn_creator',
              'mnist',
              'BasicCNNConfig',
              'KerasClassificator',
              PATH),
    Arguments('basic_cnn_creator',
              'mnist',
              'BasicCNNConfig',
              'KerasClassificator',
              PATH)
]

run_multiple(experiment_list, num_gpus=1)
