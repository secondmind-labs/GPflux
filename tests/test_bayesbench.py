import os
import numpy as np
import importlib.util
import pytest
import gpflow
import tensorflow as tf


def import_module_from_path(path, module_name):
    """Import a module based on the path as opposed to the module name"""
    if os.path.isdir(path):
        path = os.path.join(path, f"{module_name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def this_dir():
    return os.path.dirname(__file__)


@pytest.fixture
def bayesbench_deepgp():
    path = os.path.join(this_dir(), "../experiments/bayesian_benchmarks")
    return import_module_from_path(path, "bayesbench_deepgp")


def get_snelson_X_and_Y():
    data = np.load(os.path.join(this_dir(), "snelson1d.npz"))
    return data["X"], data["Y"]


def compute_gpr_lml(X, Y):
    model = gpflow.models.GPR(
        (X, Y), gpflow.kernels.SquaredExponential(), noise_variance=0.1
    )
    gpflow.optimizers.Scipy().minimize(
        model.training_loss, model.trainable_variables, compile=True
    )
    return -model.training_loss()


def test_bayesbench_deepgp_snelson(bayesbench_deepgp):
    tf.random.set_seed(0)
    np.random.seed(0)
    X, Y = get_snelson_X_and_Y()

    # log marginal likelihood of GPR with SquaredExponential kernel
    gpr_lml = compute_gpr_lml(X, Y)
    expected_elbo = -55.9003
    assert np.allclose(gpr_lml, expected_elbo)

    bench = bayesbench_deepgp.BayesBench_DeepGP()
    bench.Config.MAXITER = 3000
    bench.fit(X, Y)
    elbo = bench.log_pdf(X, Y)

    assert np.allclose(
        elbo, expected_elbo, rtol=0.02
    )  # 3000 steps not quite converged yet
    # with more optimisation steps could reduce rtol to 0.001

    # TODO: significant difference between runs - sometimes it hits ~ -56 after ~2000 steps
    # sometimes gets stuck in local optimum a few nats worse. Can we identify what these
    # optima are? Can we identify this after a few steps? Should we do random restarts more
    # generally in DGPs ?
