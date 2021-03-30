#
# Copyright (c) 2021 The GPflux Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
""" This module contains a set of utilities used in experiments and notebooks. """

import os


def is_continuous_integration() -> bool:
    """
    Return `True` if the code is running on continuous integration (CI) machines,
    otherwise `False`.

    ..note:: This check is based on the ``CI`` environment variable, which is set to ``True``
        by GitHub actions, CircleCI, and Jenkins. This function may not work as expected
        under other CI frameworks.
    """
    ci = os.environ.get("CI", "").lower()
    return (ci == "true") or (ci == "1")


def notebook_niter(n: int, test_n: int = 2) -> int:
    """
    Return a typically smaller number of iterations ``test_n`` if
    code is running on CI machines (see :func:`is_continuous_integration`),
    otherwise return ``n``.
    """
    return test_n if is_continuous_integration() else n


def notebook_range(n: int, test_n: int = 2) -> range:
    """
    Return a typically shorter `range` (of the length specified by ``test_n``) if
    code is running on CI machines (see :func:`is_continuous_integration`),
    otherwise return a `range` of the length specified by ``n``.
    """
    return range(notebook_niter(n, test_n))


def notebook_list(lst: list, test_n: int = 2) -> list:
    """
    Return a subset of the length specified by ``test_n`` from a list ``lst`` if the code
    is running on CI machines (see :func:`is_continuous_integration`),
    otherwise return the complete list ``lst``.
    """
    return lst[:test_n] if is_continuous_integration() else lst
