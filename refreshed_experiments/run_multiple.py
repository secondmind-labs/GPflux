import argparse
import multiprocessing
import os
import subprocess
import multiprocessing as mp
import time

from refreshed_experiments.utils import get_from_module
from refreshed_experiments.experiments import experiments_lists


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


def run_multiple(experiment_list, gpus):
    manager = multiprocessing.Manager()
    available_gpus = manager.Queue(maxsize=len(gpus))
    for gpu_id in gpus:
        available_gpus.put(gpu_id)
    with mp.Pool(len(gpus)) as pool:
        pool.starmap(run_experiment, [(exp, available_gpus) for exp in experiment_list])


def main():
    parser = argparse.ArgumentParser(
        description="""Entrypoint for running multiple experiments experiments. Run with""")

    parser.add_argument('--path', '-p',
                        help='Path to store the results.',
                        type=str,
                        required=True)

    parser.add_argument('--experiments_list', '-el',
                        help='Path to store the results.',
                        type=str,
                        required=True)
    parser.add_argument('--gpus',
                        help='Path to store the results.',
                        nargs='+',
                        type=str,
                        required=True)

    args = parser.parse_args()
    experiment_list_creator = get_from_module(args.experiments_list, experiments_lists)
    experiments_list = experiment_list_creator(args.path)
    run_multiple(experiments_list, gpus=args.gpus)


if __name__ == '__main__':
    main()
