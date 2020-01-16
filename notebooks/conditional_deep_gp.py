# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import gpflow
import gpflux
import numpy as np

# from: https://github.com/hughsalimbeni/bayesian_benchmarks
from bayesian_benchmarks.models import RegressionModel
from bayesian_benchmarks.tasks.regression import run as run_regression

# %%
from scipy.cluster.vq import kmeans2

def init_inducing_points(X, num):
    if X.shape[0] > num:
        return kmeans2(X, num, minit='points')[0]
    else:
        return np.concatenate([X, np.random.randn(num - X.shape[0], X.shape[1])], 0)


# %%
class ConditionalLatentDeepGP_RegressionModel(RegressionModel):
    """
    We wrap our Deep GP model in a RegressionModel class, to comply with
    bayesian_benchmarks' interface. This means we need to implement:
    - fit
    - predict
    - sample
    """
    def __init__(self, is_test=False, seed=0):
        super().__init__(is_test=is_test, seed=seed)
    
    def fit(self, X, Y):
        print("X shape:", X.shape)
        print("Y shape:", Y.shape)

        class Config:
            LATENT_DIM = 2
            X_dim, Y_dim = X.shape[1], Y.shape[1]
            D_in = X_dim + LATENT_DIM
            OUTPUT_DIMS = [D_in, Y.shape[1]]
            ADAM_LR = 0.01
            if self.is_test:
                M = 5
                MAXITER = 10
            else:
                M = 500
                MAXITER = int(10e3)
            
        print(Config.MAXITER)
        
        # Encder
        encoder = gpflux.GPflowEncoder(Config.X_dim + Config.Y_dim, Config.LATENT_DIM, [50, 50])
            
        # Layer 1
        Z1 = init_inducing_points(X, Config.M)
        Z1 = np.concatenate([Z1, np.random.randn(Z1.shape[0], Config.LATENT_DIM)], 1)
        feat1 = gpflow.features.InducingPoints(Z1)
        kern1 = gpflow.kernels.RBF(Config.D_in, lengthscales=float(Config.D_in) ** 0.5, variance=0.1)
        mean1 = gpflow.mean_functions.Identity(Config.D_in)
        layer1 = gpflux.layers.GPLayer(kern1, feat1, Config.OUTPUT_DIMS[0])

        # Layer 2
        Z2 = np.random.randn(Config.M, Config.D_in)
        feat2 = gpflow.features.InducingPoints(Z2)
        kern2 = gpflow.kernels.RBF(Config.D_in, lengthscales=float(Config.D_in) ** 0.5)
        mean2 = gpflow.mean_functions.Linear(np.random.randn(Config.D_in, Config.Y_dim))
        layer2 = gpflux.layers.GPLayer(kern2, feat2, Config.OUTPUT_DIMS[0])

        self.model = gpflux.ConditionalLatentDeepGP(X, Y, encoder, [layer1, layer2])

        # minimize
        print("before optimization:", self.model.compute_log_likelihood())
        gpflow.training.AdamOptimizer(Config.ADAM_LR).minimize(self.model, maxiter=Config.MAXITER)
        print("after optimization:", self.model.compute_log_likelihood())
    
    def predict(self, X):
        return self.model.decode(X)
    
    def sample(self, X, num_samples):
        m, v = self.model.decode(X)
        return m + np.random.randn(*m.shape) * np.sqrt(v)


# %%
class ARGS:
    seed = 0
    dataset = "energy"
    split = 0


# %%
run_regression(ARGS, is_test=True, write_to_database=False, Model=ConditionalLatentDeepGP_RegressionModel)

# %%

# %%
