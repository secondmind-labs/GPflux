"""
Optimization-related modules, currently just contains the `NatGradModel`
and `NatGradWrapper` classes to integrate
`gpflow.optimizers.NaturalGradient` with Keras.
"""
from gpflux.optimization.keras_natgrad import NatGradModel, NatGradWrapper
