import os


def is_running_pytest():
    return "RUNNING_PYTEST" in os.environ
