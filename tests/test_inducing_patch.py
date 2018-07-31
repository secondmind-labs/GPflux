# Copyright (C) PROWLER.io 2018 - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

import pytest
import numpy as np

from gpflux.convolution import InducingPatch


class Data:
    N, w, h = 10, 3, 3


@pytest.fixture
def inducing_patch():
    Z = np.random.randn(Data.N, Data.w, Data.h)
    return InducingPatch(Z)


def test_patch_shape(inducing_patch):
    actual_shape = inducing_patch.Z.read_value().shape
    desired_shape = (Data.N , Data.w * Data.h)
    assert actual_shape == desired_shape


def test_patch_len(inducing_patch):
    assert len(inducing_patch) == Data.N


def test_patch_outpus(inducing_patch):
    assert inducing_patch.outputs == 1

