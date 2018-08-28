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

def plot_latents(X, Y, encoder)
    import matplotlib.pyplot as plt
