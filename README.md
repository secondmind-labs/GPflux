# GPflux

A library built on top of GPflow to facilitate the development of Deep GPs.
Gives a user access to layers, initialization routines and Deep GP models.
This repo does not contain any new research.
It is an initiative to combine existing approaches and techniques in a single project and provide an easy, yet flexible, interface.

## Goals

- Find good initialisation practices for hyperparameters.
- Out-of-the-box working GP-Layers with minimal configuration. An expert user, however, can fine-tune the parameters, kernel, features as much as desired.
- Examples of existing Deep GP models implemented in GPflux/GPflow.

## Example

```python
import gpflow
import gpflux

X, Y = data()
D_in = X.shape[1]
D_out = Y.shape[1]

# Layer 1
Z1 = X.copy()
feat1 = gpflow.features.InducingPoints(Z1)
kern1 = gpflow.kernels.RBF(D_in)
mean1 = gpflow.mean_functions.Identity(D_in)
layer1 = gpflux.layers.GPLayer(kern1, feat1, D_in, mean_function=mean1)

# Layer 2
Z2 = X.copy()
feat2 = gpflow.features.InducingPoints(Z2)
kern2 = gpflow.kernels.RBF(D_in)
mean2 = gpflow.mean_functions.Identity(D_in)
layer2 = gpflux.layers.GPLayer(kern2, feat2, D_out, mean_function=mean2)

model = gpflux.DeepGP(X, Y, [layer1, layer2])
```

## Install

GPFlux requires TensorFlow and GPFlow to be installed.
Running the following the command adds GPFlux to your Python env.
```bash
$ python setup.py develop
```

## References

[1] Salimbeni, Hugh, and Marc Deisenroth. "Doubly stochastic variational inference for deep gaussian processes." Advances in Neural Information Processing Systems. 2017.

[2] van der Wilk, Mark, Carl Edward Rasmussen, and James Hensman. "Convolutional Gaussian Processes." Advances in Neural Information Processing Systems. 2017.

[3] Matthews, De G., et al. "GPflow: A Gaussian process library using TensorFlow." The Journal of Machine Learning Research 18.1 (2017): 1299-1304.
