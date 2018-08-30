import numpy as np


def normalize(X, mean=None, std=None):
    if mean is None:
        mean = np.average(X, 0)[None, :]
    if std is None:
        std = 1e-6 + np.std(X, 0)[None, :]

    return (X - mean) / std, mean, std


def plot(images):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, ...])
    plt.show()


def calculate_nll(model, Xs, N_mc=200, latent_dim=None, batchsize=100):

    if len(Xs) < batchsize:
        # handle the full test batch at once
        nll = model.nll(Xs, N_mc)
    else:
        # loop over chunks of test data
        Ns = len(Xs)
        splits = Ns // batchsize
        nlls = []
        for xs in np.array_split(Xs, splits):
            nlls.append(model.nll(xs, N_mc))

        nll = np.mean(nlls)

    print("Test NLL on {} test points: {}".format(len(Xs), nll))
    return nll


def trace(T, sess, name):
    import tensorflow as tf
    from tensorflow.python.client import timeline


    # add additional options to trace the session execution
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(T, options=options, run_metadata=run_metadata)

    # Create the Timeline object, and write it to a json file
    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open(name, 'w') as f:
        f.write(chrome_trace)

def plot_latents(model, Xs=None, n=1000):
    import matplotlib.pyplot as plt
    from observations import mnist

    encoder = model.layers[0].encoder

    (Xall, Yall), (_, _) = mnist("./data")
    Xall /= 255.

    def func():
        if Xs is None:
            indices = np.random.choice(np.arange(len(Xall)), n, replace=False)
            X = Xall[indices, ...]
            Y = Yall[indices, ...]
        else:
            X = Xs
            Y = [1]*len(Xs)
        qL_mean, qL_var = encoder.eval(X)
        fig, ax = plt.subplots(1, 1)
        Zs = model.layers[1].feature.feat.Z.read_value(model.enquire_session())
        ax.scatter(qL_mean[:, 0], qL_mean[:, 1], c=Y, s=2*(qL_var).max(axis=1))
        print("mean q(W) std: ", np.mean((qL_var)))
        ax.plot(Zs[:, 0], Zs[:, 1], 'kx', ms=4)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        return fig
    
    return func

import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow


def plot_inducing_patches(model):

    def func():
        patches = model.layers[2].feature.Z.read_value(model.enquire_session())
        vmin, vmax = patches.min(), patches.max()

        fig, axes = plt.subplots(30, 30, figsize=(10, 10))
        for patch, ax in zip(patches, axes.flat):
            im = ax.matshow(patch.reshape(5, 5), vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])

        cbar_ax = fig.add_axes([0.95, .1, .01, 0.8])
        fig.colorbar(im, cax=cbar_ax)
        return fig

    return func



def __plot_samples(model, axes=None, n=25, inner_gp=False, Ws=None):

    assert n / np.sqrt(n) == int(np.sqrt(n))

    if Ws is None:
        Ws = np.random.randn(n, model.layers[0].latent_dim)
    else:
        assert Ws.shape[0] ==n

    Xs = np.zeros([n, 0])
    
    if inner_gp:
        vals = model.decode_inner_layer(Xs, Ws)
        width = 32
    else:
        vals = model.predict_ys_with_Ws_full_output_cov(Xs, Ws)
        width = 28
    
    if axes is None:
        n = int(np.sqrt(n))
        fig, axes = plt.subplots(n, 2 * n, figsize=(28, 14))
    
    vals_min = vals.min()
    vals_max = vals.max()
    for i, ax in enumerate(axes.flat):
        ax.set_title("{:.2f}, {:.2f}".format(Ws[i, 0], Ws[i, 1]))
        im = ax.imshow(vals[i].reshape(width, width), vmin=vals_min, vmax=vals_max)
        # print(Fs[i].min(), Fs[i].max(), Ps[i].min(), Ps[i].max())
        ax.set_xticks([])
        ax.set_yticks([])
    
    return im


def plot_samples(model):

    def func():
        fig, axes = plt.subplots(5, 2 * 5, figsize=(20, 10))

        ww = np.linspace(-2, 2, 5)
        Ws = np.vstack([x.flatten() for x in np.meshgrid(ww, ww)]).T  # Px2
        # Ws = np.random.randn(25, model.layers[0].latent_dim)
        im1 = __plot_samples(model, axes=axes[:, :5], n=25, inner_gp=True, Ws=Ws)
        im2 = __plot_samples(model, axes=axes[:, 5:], n=25, inner_gp=False, Ws=Ws)

        fig.subplots_adjust(left=.05, right=.95)
        cbar_ax = fig.add_axes([0., .1, .01, 0.8])
        fig.colorbar(im1, cax=cbar_ax)
        cbar_ax2 = fig.add_axes([0.95, .1, .01, 0.8])
        fig.colorbar(im2, cax=cbar_ax2)
        return fig

    return func