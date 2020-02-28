# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import os


def is_continuous_integration():
    ci = os.environ.get("CI", "").lower()
    return (ci == "true") or (ci == "1")


def notebook_niter(n, test_n=2):
    return test_n if is_continuous_integration() else n


def notebook_range(n, test_n=2):
    return range(notebook_niter(n, test_n))


def notebook_list(lst, test_n=2):
    return lst[:test_n] if is_continuous_integration() else lst
