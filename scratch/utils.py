import numpy as np


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


def get_error_cb(model, Xs, Ys, error_func, full=False, Ns=500):
    def error_cb(*args, **kwargs):
        if full:
            xs, ys = Xs, Ys
        else:
            xs, ys = Xs[:Ns], Ys[:Ns]
        return error_func(model, xs, ys, batchsize=50)
    return error_cb


def trace(model, name):
    from tensorflow.python.client import timeline

    sess = model.enquire_session()
    # adam_opt = gpflow.train.AdamOptimizer(learning_rate=0.01)
    # adam_step = adam_opt.make_optimize_tensor(model, session=sess)
    like = model.likelihood_tensor

    with sess:
        # add additional options to trace the session execution
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(like, options=options, run_metadata=run_metadata)

        # Create the Timeline object, and write it to a json file
        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
        chrome_trace = fetched_timeline.generate_chrome_trace_format()
        with open(name, 'w') as f:
            f.write(chrome_trace)
# import inspect

# def generate_experiment_name(f):
#     """
#     Generates a string based on the name of the arguments
#     and the value they have.
#     Eg.

#     ```
#     @generate_experiment_name
#     def experiment_name(a, b, c):
#         pass

#     experiment_name(4, 6, 7)
#     # returns: a_4_b_6_c_7
#     ```
#     """

#     def wrapper(*args, **kwargs):

#         args_name = inspect.getargspec(f)[0]
#         name_and_args = []
#         for k, v in zip(args_name, args):
#             name_and_args.append(str(k))
#             name_and_args.append(str(v))

#         return "_".join(name_and_args)

#     return wrapper
