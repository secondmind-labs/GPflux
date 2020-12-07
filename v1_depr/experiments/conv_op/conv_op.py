import itertools as it

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.image import extract_patches_2d

from gpflux.conv_square_dists import diag_conv_inner_prod


def patch_inner_product(X, patch_shape):
    """
    Calculates the inner product between every pair of patches
    for all images of X.  Assume that X[n] is an image of
    size H x W and X[n][p] is the p-th patch in the n-th image,
    a patch has size h x w. A single pixel in the patch is
    given by X[n][p](i, j), then this function returns a matrix
    of size N x P x P:
    ```
    output[n, p, p'] = sum_{i,j} X[n, p](i,j) X[n, p'](i,j)
    ```
    where 0 <= i < patch_shape[0], 0 <= j < patch_shape[1] and
    P is the number of patches in an image.

    Note:
    ----
    In this implemention we explicitly extract the patches
    from the images to calculate the inner product.
    While this is the 'easiest' solution, it is naive in
    terms of memory consumptions, and duplicate calculations.

    Args:
    ----
    :param X: A three dimensional tensor of size N x H x W,
        where N is the number of images, H is the height and
        W is the width of the image.
    :param patch_shape: A list of ints. 1-D tensor of length 2.
        The size of the patches [h, w]. E.g., [3, 3] or [5, 5].

    Return:
    ------
    A three dimensional tensor.
    """
    patches = [extract_patches_2d(im, patch_shape) for im in X]
    patches = np.array(patches)  # N x P x h x w, where P: #patches/image
    N, P = patches.shape[0], patches.shape[1]
    patches = np.reshape(patches, [N, P, np.prod(patch_shape)])  # N x P x h*w
    ret_value = np.einsum("npi,nqi->npq", patches, patches)
    return ret_value  # N x P x P


def patch_inner_product2(X, patch_shape):
    """
    Please see `patch_inner_product` for an explanation of the
    operation.

    Note:
    ----
    In this implemention we calculate the inner product by directly
    indexing the image.
    """

    def _calculate_output_shape():
        return [X.shape[1] - patch_shape[0] + 1,
                X.shape[2] - patch_shape[1] + 1]

    Hout, Wout = _calculate_output_shape()

    def _patch_slice(p):
        """
        returns indices slice for the p-th
        patch in an image.
        """
        i = p // Wout
        j = p % Wout
        return (np.s_[:],
                np.s_[i:i+patch_shape[0]],
                np.s_[j:j+patch_shape[1]])

    N = X.shape[0]
    P = Hout * Wout
    ret_value = np.empty([N, P, P])

    # >> Here we loop over the images one-by-one:
    # >> remove s_[:] from line 8
    # for n in range(N):
    #     im = X[n, ...]
    #     ret = ret_value[n, ...]
    #     for p, q in it.product(range(P), range(P)):
    #         patch_p = im[_patch_slice(p)]
    #         patch_q = im[_patch_slice(q)]
    #         ret[p, q] = np.sum(patch_p * patch_q)

    # You can directly vectorize over the images
    for p, q in it.product(range(P), range(P)):
        # 'column' through all images: N x h x w
        patch_p = X[_patch_slice(p)]
        patch_q = X[_patch_slice(q)]
        inner_prod = np.sum(patch_p * patch_q, axis=(1, 2))
        ret_value[:, p, q] = inner_prod

    return ret_value


if __name__ == "__main__":

    N = 10  # number of images
    H, W = 28, 28  # height and width of the images
    h, w = 3, 3  # height and width of the patch

    filter_shape = [h, w]
    # create dummy images
    X = np.random.randn(N, H, W)

    sess = tf.Session()
    X_1 = X[..., None]
    tf_x_dotconv = diag_conv_inner_prod(X_1, filter_shape, parallel_iterations=10, back_prop=False)

    import timeit

    t = timeit.timeit(lambda: patch_inner_product2(X, filter_shape), number=1)
    print(t)

    t = timeit.timeit(lambda: patch_inner_product(X, filter_shape), number=1)
    print(t)

    t = timeit.timeit(lambda: sess.run(tf_x_dotconv), number=1)
    print(t)

    r1 = patch_inner_product(X, filter_shape)
    r2 = patch_inner_product2(X, filter_shape)
    r3 = np.squeeze(sess.run(tf_x_dotconv))

    np.testing.assert_almost_equal(r1, r2)
    np.testing.assert_almost_equal(r1, r3)
    np.testing.assert_almost_equal(r2, r3)
