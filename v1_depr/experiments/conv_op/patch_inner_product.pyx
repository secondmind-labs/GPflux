# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False


import numpy as np
cimport numpy as np
cimport cython

from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free


FLOATTYPE = np.float64
ctypedef np.float64_t FLOATTYPE_t
INTTYPE = np.int64
ctypedef np.int64_t INTTYPE_t


@cython.cdivision(True)
def patch_inner_product(FLOATTYPE_t[:, :, ::1] X, patch_shape):
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
    In this implemention we calculate the inner product by directly
    indexing the image.

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

    cdef int Hin = X.shape[1]
    cdef int Win = X.shape[2]
    cdef int p0 = patch_shape[0]
    cdef int p1 = patch_shape[1]
    cdef int Hout = Hin - p0 + 1
    cdef int Wout = Win - p1 + 1
    cdef int Pin = Hin * Win

    cdef int N = X.shape[0]
    cdef int P = Hout * Wout

    cdef FLOATTYPE_t* ret_value = \
            <FLOATTYPE_t*> malloc(sizeof(FLOATTYPE_t) * N * P**2)

    cdef int p, q, pi, pj, qi, qj, n

    cdef FLOATTYPE_t* ptr = <FLOATTYPE_t*> (&X[0, 0, 0])

    cdef int idx_p, idx_q

    with nogil, parallel():
        for n in prange(N):
            for p in range(P):
                pi = p // Wout
                pj = p % Wout
                for q in range(P):
                    if p <= q:
                        qi = q // Wout
                        qj = q % Wout

                        idx_p = n * Pin + pi * Win + pj
                        idx_q = n * Pin + qi * Win + qj
                        out = ret_value + n * P**2 + p * P + q
                        _inner_prod_matrices(ptr + idx_p, ptr + idx_q, p0, p1, Win, out)
                    else:
                        out = ret_value + n * P**2 + p * P + q
                        _in = ret_value + n * P**2 + q * P + p
                        out[0] = _in[0]


    return <FLOATTYPE_t[:N, :P, :P]> ret_value


cdef inline void _inner_prod_matrices(FLOATTYPE_t* patch_p,
                                      FLOATTYPE_t* patch_q,
                                      int p0, int p1, int Win,
                                      FLOATTYPE_t* out) nogil:

    cdef FLOATTYPE_t v = 0
    cdef int i, j

    for i in range(p0):
        for j in range(p1):
            v += patch_p[i*Win + j] * patch_q[i*Win + j]

    out[0] = v
