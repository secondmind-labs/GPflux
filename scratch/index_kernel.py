import tensorflow as tf
import numpy as np
import gpflux
import gpflow
import gpflow.training.monitor as mon
import logging

from data import mnist, mnist01

gpflux.conditionals.logger.setLevel(logging.DEBUG)

def plot(images):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, ...])
    plt.show()

def calc_binary_error(model, Xs, Ys, batchsize=100):
    Ns = len(Xs)
    splits = Ns // batchsize
    hits = []
    for xs, ys in zip(np.array_split(Xs, splits), np.array_split(Ys, splits)):
        p, _ = model.predict_y(xs)
        acc = ((p > 0.5).astype('float') == ys)
        hits.append(acc)
    error = 1.0 - np.concatenate(hits, 0)
    return np.sum(error) * 100.0 / len(error)

def calc_multiclass_error(model, Xs, Ys, batchsize=100):
    Ns = len(Xs)
    splits = Ns // batchsize
    hits = []
    for xs, ys in zip(np.array_split(Xs, splits), np.array_split(Ys, splits)):
        p, _ = model.predict_y(xs)
        acc = p.argmax(1) == ys[:, 0]
        hits.append(acc)
    error = 1.0 - np.concatenate(hits, 0)
    return np.sum(error) * 100.0 / len(error)


Nc = 2
if Nc > 2:
    print("Multiclass")
    X, Y, Xs, Ys = mnist()
    X = X + np.random.randn(*X.shape) * 1e-3
    Y = Y.astype(int)
    Ys = Ys.astype(int)
else:
    print("Binary")
    X, Y, Xs, Ys = mnist01()

print(X.shape)
print(Y.shape)
print(Xs.shape)
print(Ys.shape)

# plot(X.reshape(-1, 28, 28)[:100, ...])
# print(Y[:100])
# exit(0)

N = X.shape[0]
M = 20
H, W = 28, 28
assert H == W
patch_size = np.array([5, 5])
img_size_in = np.array([H, W])

base_kernel = gpflow.kernels.RBF(np.prod(patch_size)) # + gpflow.kernels.White(np.prod(patch_size), variance=0.01)
conv_kernel = gpflux.convolution_kernel.ConvKernel(base_kernel, img_size_in, patch_size)
index_kernel = gpflow.kernels.RBF(2, lengthscales=5., ARD=True) # + gpflow.kernels.White(2, variance=0.01)
kern = gpflux.convolution_kernel.IndexedConvKernel(conv_kernel, index_kernel)

inducing_patches_initializer = gpflux.init.PatchSamplerInitializer(X, width=W, height=H)
inducing_patches = inducing_patches_initializer.sample([M, *patch_size])  # M x w x h
inducing_patches = gpflux.inducing_patch.InducingPatch(inducing_patches.reshape([M, np.prod(patch_size)]))


Z_indices = np.random.randint(0, H, size=(M, 2))
inducing_indices = gpflow.features.InducingPoints(Z_indices)
feat = gpflux.inducing_patch.IndexedInducingPatch(inducing_patches, inducing_indices)

calc_error = calc_binary_error if Nc == 2 else calc_multiclass_error

if Nc > 2:
    like = gpflow.likelihoods.SoftMax(Nc)
    num_latent = Nc
else:
    like = gpflow.likelihoods.Bernoulli()
    num_latent = 1
print(like.__class__.__name__)
m = gpflow.models.SVGP(X, Y, kern,
                       likelihood=like,
                       feat=feat,
                       num_latent=num_latent,
                       minibatch_size=200)
m.q_sqrt = m.q_sqrt.read_value() * 1e-3

MAXITER = 1000
print_task = mon.PrintTimingsTask().with_name('print')\
    .with_condition(mon.PeriodicIterationCondition(10))\
    .with_exit_condition(True)

def cb(*args, **kwargs):
    # m.anchor(m.enquire_session())
    # print(m)
    print("elbo", m.compute_log_likelihood())

print_lml = mon.CallbackTask(cb)\
    .with_name('lml')\
    .with_condition(mon.PeriodicIterationCondition(50))\
    .with_exit_condition(True)

def cb2(*args, **kwargs):
    Ns = 1000
    print("error", calc_error(m, Xs[:Ns], Ys[:Ns], batchsize=50))

print_error = mon.CallbackTask(cb2)\
    .with_name('error')\
    .with_condition(mon.PeriodicIterationCondition(50))\
    .with_exit_condition(True)

session = m.enquire_session()
global_step = mon.create_global_step(session)

print("before", m.compute_log_likelihood())
print("before error", calc_error(m, Xs, Ys))

optimiser = gpflow.train.AdamOptimizer(0.01)

monitor = mon.Monitor([print_task, print_lml, print_error], session, global_step, print_summary=True)
with monitor:
    optimiser.minimize(m, step_callback=monitor, maxiter=MAXITER, global_step=global_step)

print("after", m.compute_log_likelihood())
print("after error", calc_error(m, Xs, Ys))

print(m)
