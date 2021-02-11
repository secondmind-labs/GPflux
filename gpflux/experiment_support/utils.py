# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import os


def is_continuous_integration() -> bool:
    ci = os.environ.get("CI", "").lower()
    return (ci == "true") or (ci == "1")


def notebook_niter(n: int, test_n: int = 2) -> int:
    return test_n if is_continuous_integration() else n


def notebook_range(n: int, test_n: int = 2) -> range:
    return range(notebook_niter(n, test_n))


def notebook_list(lst: list, test_n: int = 2) -> list:
    return lst[:test_n] if is_continuous_integration() else lst
