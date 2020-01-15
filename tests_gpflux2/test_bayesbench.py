import os
import numpy as np
import importlib.util
import pytest

def this_dir():
    return os.path.dirname(__file__)

def get_bayesbench_experiment_directory():
    return os.path.join(this_dir(), '../experiments/bayesian_benchmarks')

@pytest.fixture
def bayesbench_deepgp():
    path = os.path.join(get_bayesbench_experiment_directory(), "bayesbench_deepgp.py")
    spec = importlib.util.spec_from_file_location("bayesbench_deepgp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_snelson_X_and_Y():
    data = np.load(os.path.join(this_dir(), 'snelson1d.npz'))
    return data['X'], data['Y']

def test_bayesbench_deepgp_snelson(bayesbench_deepgp):
    X, Y = get_snelson_X_and_Y()
    bench = bayesbench_deepgp.BayesBench_DeepGP()
    bench.Config.MAXITER = 3000
    bench.fit(X, Y)
    elbo = bench.log_pdf(X, Y)
    expected_elbo = - 55.9
    assert np.abs(elbo - expected_elbo) < 1.0
