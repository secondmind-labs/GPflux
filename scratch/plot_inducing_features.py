import numpy as np
import gpflow
import matplotlib.pyplot as plt

def plot(images):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 10, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, ...])
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


W = H = 28
model = gpflow.Saver().load("./indexed_conv_svgp_M_20.gpflow")
Z_patch = model.feature.inducing_patches.Z.read_value()
Z_idx = model.feature.inducing_indices.Z.read_value()
patch_size = model.kern.conv_kernel.patch_size[0]

Xnew = np.ones([len(model.feature), H, W]) * 100
for i in range(len(model.feature)):
    idx_height = min(H-patch_size, int(Z_idx[i, 0]))
    idx_width = min(W-patch_size, int(Z_idx[i, 1]))
    height_slice = np.s_[idx_height:idx_height+5]
    width_slice = np.s_[idx_width:idx_width+5]
    Xnew[i, height_slice, width_slice] = Z_patch[i, ...].reshape([5, 5])

f, _ = model.predict_f(Xnew.reshape(-1, H*W))
# f = f * model.kern.conv_kernel.num_patches

fig, ax = plt.subplots()
ax.set_xlim(0, 28)
ax.set_ylim(0, 28)
ax.imshow(np.ones((H, W)) * np.nan, vmin=-1, vmax=1, extent=[0, W, 0, H], cmap="Greys", origin="upper")
for i in range(len(model.feature)):
    idx_height = min(H-patch_size, int(Z_idx[i, 0]))
    idx_width = min(W-patch_size, int(Z_idx[i, 1]))
    extent=[idx_width, idx_width+5, idx_height, idx_height+5]
    cmap = "Reds" if f[i][0] > 0 else "Blues"
    print(f[i][0])
    ax.imshow(Z_patch[i, ...].reshape([5, 5]), extent=extent, cmap=cmap, origin="upper")

plt.show()
