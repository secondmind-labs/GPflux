import numpy as np

from gpflux.initializers import GivenVariationalInitializer, FeedForwardInitializer
from gpflux.helpers import construct_gp_layer


def test_givenvariationalinitializer():
    N, M, D, P = 13, 7, 5, 3
    q_mu = np.random.randn(M, P)
    q_sqrt = np.tril(np.random.randn(P, M, M))
    qu_initializer = GivenVariationalInitializer(q_mu, q_sqrt)
    initializer = FeedForwardInitializer(qu_initializer=qu_initializer)
    layer = construct_gp_layer(N, M, D, P, initializer=initializer)
    X = np.random.randn(N, D)
    _ = layer(X)
    np.testing.assert_allclose(layer.q_mu.numpy(), q_mu)
    np.testing.assert_allclose(layer.q_sqrt.numpy(), q_sqrt)
