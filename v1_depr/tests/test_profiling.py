# Copyright (C) PROWLER.io 2019 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import os
import time

from gpflux.profile import get_timing_tasks, Timer, TimingTask


def test_smoke_profiling():
    """
    Smoke test for profiling used in Makefile.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    timing_tasks = get_timing_tasks(num_optimisation_updates=1,
                                    num_warm_up=0,
                                    num_iterations=1)
    timer = Timer(task_list=timing_tasks)
    timer.run()


def test_timer():
    """
    Tests the Timer and returned resulting string.
    """

    def test_creator():
        def timed_method():
            time.sleep(0.0001)
            return 0.01

        return timed_method

    timing_task = TimingTask('test_task', test_creator, num_iterations=1, num_warm_up=0)
    timer = Timer(task_list=[timing_task])
    result = timer.run()
    assert result == 'Timings:\nTask for test_task: mean 10.000 ms, std 0.000 ms'
