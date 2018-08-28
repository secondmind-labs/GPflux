import numpy as np
import argparse
import observations

import gpflow

from observations import mnist
from utils import get_error_cb, calc_multiclass_error


def data():
    (X, Y), (Xs, Ys) = mnist("./data")
    Y = Y.astype(int)
    Ys = Ys.astype(int)
    Y = Y.reshape(-1, 1)
    Ys = Ys.reshape(-1, 1)
    alpha = 255.0
    return X/alpha, Y, Xs/alpha, Ys

def get_misclassified_images(model, Xs, Ys, batchsize=50):
    Nc = 10
    Ns = len(Xs)
    D = Xs.shape[1]
    splits = Ns // batchsize
    missclassified_X = []
    missclassified_Y = []
    missclassified_probs = []
    all_probs = []
    for idx, xs, ys in zip(np.array_split(np.arange(Ns), splits),
                           np.array_split(Xs, splits),
                           np.array_split(Ys, splits)):
        num = 10
        ns = len(xs)
        #  xs: Ns x D
        xss = np.tile(xs[None, ...], [num, 1, 1])  # num x ns x D
        xss = np.reshape(xss, [num * ns, D])  # num*ns x D
        p, _ = model.predict_y(xss)  # num*ns x Nc
        p = np.reshape(p, [num, ns, Nc])  # num x ns x Nc
        p = np.mean(p, axis=0, keepdims=False) # ns x Nc
        all_probs.append(p)
        misses = np.argmax(p, axis=1) != ys[:, 0]
        missclassified_X.append(xs[misses])
        missclassified_Y.append(ys[misses])
        missclassified_probs.append(p[misses])

    return (np.concatenate(missclassified_X, axis=0),
            np.concatenate(missclassified_Y, axis=0),
            np.concatenate(missclassified_probs, axis=0),
            np.concatenate(all_probs, axis=0))

def plot2(Xm, Pm, Ym):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(10, 20, figsize=(18, 10))
    axes = axes.flatten()
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    for i, (xm, pm, ys) in enumerate(zip(Xm, Pm, Ym)):
        ax1, ax2 = axes[2*i], axes[2*i+1]
        ax1.imshow(xm.reshape(28, 28))
        ax2.bar(np.arange(10), pm)
        ax2.bar([np.argmax(pm)], [pm.max()], color="orange")
        ax2.bar(ys, pm[ys], color="red")
        # ax2.plot(ys, .9, "rx")
        ax2.text(-0.3, .85, str(ys), color="r", fontsize=8)
        ax2.text(7.1, .85, [np.argmax(pm)], color="orange", fontsize=8)
        ax2.set_ylim(0, 1)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyse model')
    parser.add_argument('-fn', '--filename', type=str)
    args = parser.parse_args()

    X, Y, Xs, Ys = data()

    fn = args.filename
    model = gpflow.Saver().load(fn)
    model.compile()

    print("Identifying missclassified images")
    Xm, Ym_true, Pm, Pall = get_misclassified_images(model, Xs, Ys)

    print(len(Xm) / len(Xs))

    # Ym = model.predict_y(Xm)[0]
    plot2(Xm[:100], Pm[:100], Ym_true[:100])

    np.savez("weighted_conv_gp_results_new", Xm=Xm, Ym=Ym_true, Pm=Pm, Pall=Pall)
