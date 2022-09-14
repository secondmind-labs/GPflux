import os
from unittest import mock

import pytest

from gpflux.experiment_support import ci_utils


def mock_os_environ(os_env: dict):
    # without clear=True, os_env={} would have no effect
    return mock.patch.dict(os.environ, os_env, clear=True)


def mock_ci_state(is_ci: bool):
    return mock_os_environ({"CI": "true"} if is_ci else {})


@pytest.mark.parametrize(
    "os_environ, is_ci",
    [
        ({"CI": "true"}, True),  # GitHub actions, CircleCI, Travis CI
        ({"CI": "1"}, True),  # for lazy local use
        ({}, False),
        ({"CI": ""}, False),
        ({"CI": "0"}, False),
        ({"CI": "false"}, False),
    ],
)
def test_is_continuous_integration(os_environ, is_ci):
    with mock_os_environ(os_environ):
        assert ci_utils.is_continuous_integration() == is_ci


@pytest.mark.parametrize(
    "is_ci, args, expected_result", [(True, (13,), 2), (True, (13, 5), 5), (False, (13,), 13),],
)
def test_notebook_niter(is_ci, args, expected_result):
    with mock_ci_state(is_ci):
        assert ci_utils.notebook_niter(*args) == expected_result
