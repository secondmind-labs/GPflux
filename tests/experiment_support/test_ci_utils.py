#
# Copyright (c) 2022 The GPflux Contributors.
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
import os
from typing import Union

import pytest

from gpflux.experiment_support.ci_utils import (
    is_continuous_integration,
    notebook_list,
    notebook_niter,
    notebook_range,
)


class CIEnviroment:
    """Context manager to simulate a set up where a CI env variable is used"""

    def __init__(self, flag: Union[str, bool]) -> None:
        """
        :param flag: the CI env variable value
        """
        if isinstance(flag, str):
            self._flag = False if flag == "" or flag == "0" else True
        else:
            self._flag = flag

        try:
            self._ci = os.environ["CI"]
        except KeyError:
            self._ci = "false"

    def __enter__(self):
        os.environ["CI"] = str(self._flag).lower()

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.environ["CI"] = self._ci


@pytest.mark.parametrize("ci", [True, False, "1", "0", ""])
def test_is_continuous_integration(ci: bool) -> None:
    with CIEnviroment(ci):
        if isinstance(ci, str):
            ci = False if ci == "" or ci == "0" else True
        assert is_continuous_integration() == ci


@pytest.mark.parametrize("ci,niter", [(True, 2), (False, 10)])
def test_notebook_niter(ci: bool, niter: int) -> None:
    with CIEnviroment(ci):
        assert notebook_niter(10) == niter


@pytest.mark.parametrize("ci,niter", [(True, 2), (False, 10)])
def test_notebook_range(ci: bool, niter: int) -> None:
    with CIEnviroment(ci):
        assert notebook_range(10) == range(niter)


@pytest.mark.parametrize("ci,niter", [(True, 2), (False, 10)])
def test_notebook_list(ci: bool, niter: int) -> None:
    with CIEnviroment(ci):
        assert list(range(niter)) == notebook_list(list(range(10)))
