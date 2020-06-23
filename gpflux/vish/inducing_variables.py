import gpflow
import tensorflow as tf

from gpflux.vish.spherical_harmonics import SphericalHarmonicsCollection


class SphericalHarmonicInducingVariable(
    gpflow.inducing_variables.InducingVariables
):
    """Wrapper that contains the spherical harmonics"""

    def __init__(self, harmonics: SphericalHarmonicsCollection):
        self.harmonic = harmonics

    def __call__(self, X):
        return self.harmonic(X)

    def __len__(self):
        return len(self.harmonic)

    def num_levels(self):
        return self.harmonic.num_levels()
