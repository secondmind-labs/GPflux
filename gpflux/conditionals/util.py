from typing import Optional, Tuple

import tensorflow as tf

from gpflow.base import MeanAndVariance
from gpflow.conditionals.util import rollaxis_left
from gpflow.config import default_float


def base_orthogonal_conditional(
    Kmn: tf.Tensor,
    Kmm: tf.Tensor,
    Knn: tf.Tensor,
    Cmn: tf.Tensor,
    Cmm: tf.Tensor,
    Cnn: tf.Tensor,
    f_u: tf.Tensor,
    f_v: tf.Tensor,
    *,
    full_cov: bool = False,
    q_sqrt_u: Optional[tf.Tensor] = None,
    q_sqrt_v: Optional[tf.Tensor] = None,
    white: bool = False,
    Lm: Optional[tf.Tensor] = None,
) -> MeanAndVariance:
    r"""

    #TODO -- this needs to be updated to suit sparse orthogonal GPs
    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2; 0, Kmm)

      p(g1) = N(g1; 0, Knn)
      p(g1 | g2) = N(g1; Knm (Kmm⁻¹) g2, Knn - Knm (Kmm⁻¹) Kmn)

    And
      q(g2) = N(g2; f, q_sqrt q_sqrtᵀ)

    This method computes the mean and (co)variance of
      q(g1) = ∫ q(g2) p(g1 | g2)

    :param Kmn: [M, ..., N]
    :param Kmm: [M, M]
    :param Knn: [..., N, N]  or  N
    :param f: [M, R]
    :param full_cov: bool
    :param q_sqrt: If this is a Tensor, it must have shape [R, M, M] (lower
        triangular) or [M, R] (diagonal)
    :param white: bool
    :return: [N, R]  or [R, N, N]
    """

    # NOTE -- this is now passed from _get_Cnn method of posterior class
    # hence the Kmm argumnent is a bit redundant now
    # Lm = tf.linalg.cholesky(Kmm)
    return base_orthogonal_conditional_with_lm(
        Kmn=Kmn,
        Lm=Lm,
        Knn=Knn,
        L_Cmm=tf.linalg.cholesky(Cmm),
        Cmn=Cmn,
        Cnn=Cnn,
        f_u=f_u,
        f_v=f_v,
        full_cov=full_cov,
        q_sqrt_u=q_sqrt_u,
        q_sqrt_v=q_sqrt_v,
        white=white,
    )


def base_orthogonal_conditional_with_lm(
    Kmn: tf.Tensor,
    Lm: tf.Tensor,
    Knn: tf.Tensor,
    L_Cmm: tf.Tensor,
    Cmn: tf.Tensor,
    Cnn: tf.Tensor,
    f_u: tf.Tensor,
    f_v: tf.Tensor,
    *,
    full_cov: bool = False,
    q_sqrt_u: Optional[tf.Tensor] = None,
    q_sqrt_v: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    r"""
    Has the same functionality as the `base_conditional` function, except that instead of
    `Kmm` this function accepts `Lm`, which is the Cholesky decomposition of `Kmm`.

    This allows `Lm` to be precomputed, which can improve performance.
    """
    # compute kernel stuff
    num_func = tf.shape(f_u)[-1]  # R
    N = tf.shape(Kmn)[-1]
    M = tf.shape(f_u)[-2]
    V = tf.shape(f_v)[-2]

    # get the leading dims in Kmn to the front of the tensor
    # if Kmn has rank two, i.e. [M, N], this is the identity op.
    K = tf.rank(Kmn)
    perm = tf.concat(
        [
            tf.reshape(tf.range(1, K - 1), [K - 2]),  # leading dims (...)
            tf.reshape(0, [1]),  # [M]
            tf.reshape(K - 1, [1]),
        ],
        0,
    )  # [N]
    Kmn = tf.transpose(Kmn, perm)  # [..., M, N]
    Cmn = tf.transpose(Cmn, perm)  # [..., M, N]

    shape_constraints = [
        (Kmn, [..., "M", "N"]),
        (Lm, ["M", "M"]),
        (Knn, [..., "N", "N"] if full_cov else [..., "N"]),
        (f_u, ["M", "R"]),
        (Cmn, [..., "V", "N"]),
        (L_Cmm, ["V", "V"]),
        (Cnn, [..., "N", "N"] if full_cov else [..., "N"]),
        (f_v, ["V", "R"]),
    ]
    if q_sqrt_u is not None:
        shape_constraints.append(
            (q_sqrt_u, (["M", "R"] if q_sqrt_u.shape.ndims == 2 else ["R", "M", "M"]))
        )

    if q_sqrt_v is not None:
        shape_constraints.append(
            (q_sqrt_v, (["V", "R"] if q_sqrt_v.shape.ndims == 2 else ["R", "V", "V"]))
        )

    tf.debugging.assert_shapes(
        shape_constraints,
        message="base_orthogonal_conditional() arguments "
        "[Note that this check verifies the shape of an alternative "
        "representation of Kmn. See the docs for the actual expected "
        "shape.]",
    )

    leading_dims = tf.shape(Kmn)[:-2]

    ###################################################################################

    fmean_u, fvar_u = conditional_GP_maths(
        leading_dims=leading_dims,
        Lm=Lm,
        Kmn=Kmn,
        Knn=Knn,
        num_func=num_func,
        M=M,
        N=N,
        f=f_u,
        q_sqrt=q_sqrt_u,
        white=white,
        full_cov=full_cov,
        just_parametric=True,
    )

    fmean_v, fvar_v = conditional_GP_maths(
        leading_dims=leading_dims,
        Lm=L_Cmm,
        Kmn=Cmn,
        Knn=Cnn,
        num_func=num_func,
        M=V,
        N=N,
        f=f_v,
        q_sqrt=q_sqrt_v,
        white=white,
        full_cov=full_cov,
        just_parametric=False,
    )

    ###################################################################################

    shape_constraints = [
        (Kmn, [..., "M", "N"]),  # tensor included again for N dimension
        (f_u, [..., "M", "R"]),  # tensor included again for R dimension
        (fmean_u, [..., "N", "R"]),
        (fvar_u, [..., "R", "N", "N"] if full_cov else [..., "N", "R"]),
        (fmean_v, [..., "N", "R"]),
        (fvar_v, [..., "R", "N", "N"] if full_cov else [..., "N", "R"]),
    ]
    tf.debugging.assert_shapes(
        shape_constraints, message="base_orthogonal_)conditional() return values"
    )

    return fmean_u + fmean_v, fvar_u + fvar_v


def base_heteroskedastic_orthogonal_conditional(
    Kmn: tf.Tensor,
    Kmm: tf.Tensor,
    Knn: tf.Tensor,
    Cmn: tf.Tensor,
    Cmm: tf.Tensor,
    Cnn: tf.Tensor,
    f_u: tf.Tensor,
    f_v: tf.Tensor,
    *,
    full_cov: bool = False,
    q_sqrt_u: Optional[tf.Tensor] = None,
    q_sqrt_v: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    r"""

    #TODO -- this needs to be updated to suit sparse orthogonal GPs
    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2; 0, Kmm)

      p(g1) = N(g1; 0, Knn)
      p(g1 | g2) = N(g1; Knm (Kmm⁻¹) g2, Knn - Knm (Kmm⁻¹) Kmn)

    And
      q(g2) = N(g2; f, q_sqrt q_sqrtᵀ)

    This method computes the mean and (co)variance of
      q(g1) = ∫ q(g2) p(g1 | g2)

    :param Kmn: [M, ..., N]
    :param Kmm: [M, M]
    :param Knn: [..., N, N]  or  N
    :param f: [M, R]
    :param full_cov: bool
    :param q_sqrt: If this is a Tensor, it must have shape [R, M, M] (lower
        triangular) or [M, R] (diagonal)
    :param white: bool
    :return: [N, R]  or [R, N, N]
    """
    Lm = tf.linalg.cholesky(Kmm)
    return base_heteroskedastic_orthogonal_conditional_with_lm(
        Kmn=Kmn,
        Lm=Lm,
        Knn=Knn,
        L_Cmm=tf.linalg.cholesky(Cmm),
        Cmn=Cmn,
        Cnn=Cnn,
        f_u=f_u,
        f_v=f_v,
        full_cov=full_cov,
        q_sqrt_u=q_sqrt_u,
        q_sqrt_v=q_sqrt_v,
        white=white,
    )


def base_heteroskedastic_orthogonal_conditional_with_lm(
    Kmn: tf.Tensor,
    Lm: tf.Tensor,
    Knn: tf.Tensor,
    L_Cmm: tf.Tensor,
    Cmn: tf.Tensor,
    Cnn: tf.Tensor,
    f_u: tf.Tensor,
    f_v: tf.Tensor,
    *,
    full_cov: bool = False,
    q_sqrt_u: Optional[tf.Tensor] = None,
    q_sqrt_v: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    r"""
    Has the same functionality as the `base_conditional` function, except that instead of
    `Kmm` this function accepts `Lm`, which is the Cholesky decomposition of `Kmm`.

    This allows `Lm` to be precomputed, which can improve performance.
    """
    # compute kernel stuff
    num_func = tf.shape(f_u)[-1]  # R
    N = tf.shape(Kmn)[-1]
    M = tf.shape(f_u)[-2]

    # get the leading dims in Kmn to the front of the tensor
    # if Kmn has rank two, i.e. [M, N], this is the identity op.
    K = tf.rank(Kmn)
    perm = tf.concat(
        [
            tf.reshape(tf.range(1, K - 1), [K - 2]),  # leading dims (...)
            tf.reshape(0, [1]),  # [M]
            tf.reshape(K - 1, [1]),
        ],
        0,
    )  # [N]
    Kmn = tf.transpose(Kmn, perm)  # [..., M, N]
    Cmn = tf.transpose(Cmn, perm)  # [..., M, N]

    shape_constraints = [
        (Kmn, [..., "M", "N"]),
        (Lm, ["M", "M"]),
        (Knn, [..., "N", "N"] if full_cov else [..., "N"]),
        (f_u, ["M", "R"]),
        (Cmn, [..., "V", "N"]),
        (L_Cmm, ["V", "V"]),
        (Cnn, [..., "N", "N"] if full_cov else [..., "N"]),
        (f_v, ["V", "R"]),
    ]
    if q_sqrt_u is not None:
        shape_constraints.append(
            (q_sqrt_u, (["M", "R"] if q_sqrt_u.shape.ndims == 2 else ["R", "M", "M"]))
        )

    if q_sqrt_v is not None:
        shape_constraints.append(
            (q_sqrt_v, (["V", "R"] if q_sqrt_v.shape.ndims == 2 else ["R", "V", "V"]))
        )

    tf.debugging.assert_shapes(
        shape_constraints,
        message="base_orthogonal_conditional() arguments "
        "[Note that this check verifies the shape of an alternative "
        "representation of Kmn. See the docs for the actual expected "
        "shape.]",
    )

    leading_dims = tf.shape(Kmn)[:-2]

    ###################################################################################

    fmean_u, fvar_u = conditional_GP_maths(
        leading_dims=leading_dims,
        Lm=Lm,
        Kmn=Kmn,
        Knn=Knn,
        num_func=num_func,
        M=M,
        N=N,
        f=f_u,
        q_sqrt=q_sqrt_u,
        white=white,
        full_cov=full_cov,
        just_parametric=True,
    )

    fmean_v, fvar_v = conditional_GP_maths(
        leading_dims=leading_dims,
        Lm=L_Cmm,
        Kmn=Cmn,
        Knn=Cnn,
        num_func=num_func,
        M=tf.shape(f_v)[-2],
        N=N,
        f=f_v,
        q_sqrt=q_sqrt_v,
        white=white,
        full_cov=full_cov,
        just_parametric=False,
    )

    ###################################################################################

    shape_constraints = [
        (Kmn, [..., "M", "N"]),  # tensor included again for N dimension
        (f_u, [..., "M", "R"]),  # tensor included again for R dimension
        (fmean_u, [..., "N", "R"]),
        (fvar_u, [..., "R", "N", "N"] if full_cov else [..., "N", "R"]),
        (fmean_v, [..., "N", "R"]),
        (fvar_v, [..., "R", "N", "N"] if full_cov else [..., "N", "R"]),
    ]
    tf.debugging.assert_shapes(
        shape_constraints, message="base_orthogonal_)conditional() return values"
    )

    return tf.concat([fmean_u, fmean_v], axis=-1), tf.concat([fvar_u, fvar_v], axis=-1)


def conditional_GP_maths(  # noqa: C901
    leading_dims: tf.Tensor,
    Lm: tf.Tensor,
    Kmn: tf.Tensor,
    Knn: tf.Tensor,
    num_func: int,
    M: int,
    N: int,
    f: tf.Tensor,
    q_sqrt: tf.Tensor,
    just_parametric: bool,
    white: bool = True,
    full_cov: bool = False,
) -> MeanAndVariance:

    # Compute the projection matrix A
    Lm = tf.broadcast_to(Lm, tf.concat([leading_dims, tf.shape(Lm)], 0))  # [..., M, M]
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [..., M, N]

    if just_parametric:
        pass
    else:

        # compute the covariance due to the conditioning
        if full_cov:
            fvar = Knn - tf.linalg.matmul(A, A, transpose_a=True)  # [..., N, N]
            cov_shape = tf.concat([leading_dims, [num_func, N, N]], 0)
            fvar = tf.broadcast_to(tf.expand_dims(fvar, -3), cov_shape)  # [..., R, N, N]
        else:
            fvar = Knn - tf.reduce_sum(tf.square(A), -2)  # [..., N]
            cov_shape = tf.concat([leading_dims, [num_func, N]], 0)  # [..., R, N]
            fvar = tf.broadcast_to(tf.expand_dims(fvar, -2), cov_shape)  # [..., R, N]

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)

    # construct the conditional mean
    f_shape = tf.concat([leading_dims, [M, num_func]], 0)  # [..., M, R]
    f = tf.broadcast_to(f, f_shape)  # [..., M, R]
    fmean = tf.linalg.matmul(A, f, transpose_a=True)  # [..., N, R]

    if q_sqrt is not None:
        q_sqrt_dims = q_sqrt.shape.ndims
        if q_sqrt_dims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # [R, M, N]
        elif q_sqrt_dims == 3:
            L = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle # [R, M, M]
            L_shape = tf.shape(L)
            L = tf.broadcast_to(L, tf.concat([leading_dims, L_shape], 0))

            shape = tf.concat([leading_dims, [num_func, M, N]], axis=0)
            A_tiled = tf.broadcast_to(tf.expand_dims(A, -3), shape)
            LTA = tf.linalg.matmul(L, A_tiled, transpose_a=True)  # [R, M, N]
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.shape.ndims))

        # NOTE -- very ineffective at the moment
        if just_parametric:

            if full_cov:
                fvar = tf.linalg.matmul(LTA, LTA, transpose_a=True)  # [R, N, N]
            else:
                fvar = tf.reduce_sum(tf.square(LTA), -2)  # [R, N]

        else:

            if full_cov:
                fvar = fvar + tf.linalg.matmul(LTA, LTA, transpose_a=True)  # [R, N, N]
            else:
                fvar = fvar + tf.reduce_sum(tf.square(LTA), -2)  # [R, N]

    if not full_cov:
        fvar = tf.linalg.adjoint(fvar)  # [N, R]

    ####################################################################

    return fmean, fvar


def separate_independent_orthogonal_conditional_implementation(
    Kmns: tf.Tensor,
    Kmms: tf.Tensor,
    Knns: tf.Tensor,
    Cmns: tf.Tensor,
    Cmms: tf.Tensor,
    Cnns: tf.Tensor,
    f_u: tf.Tensor,
    f_v: tf.Tensor,
    *,
    full_cov: bool = False,
    q_sqrt_u: Optional[tf.Tensor] = None,
    q_sqrt_v: Optional[tf.Tensor] = None,
    white: bool = False,
    Lms: Optional[tf.Tensor] = None,
) -> MeanAndVariance:
    """
    Multi-output GP with independent GP priors.

    Number of latent processes equals the number of outputs (L = P).

    Further reference:

    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multioutput notebook for more information about the multioutput framework.
    - See above for the parameters and the return value.
    """
    fs_u = tf.transpose(f_u)[:, :, None]  # [P, M_u, 1]
    # [P, 1, M_u, M_u]  or  [P, M_u, 1]

    fs_v = tf.transpose(f_v)[:, :, None]  # [P, M_v, 1]
    # [P, 1, M_v, M_v]  or  [P, M_v, 1]

    if q_sqrt_u is not None and q_sqrt_v is not None:
        q_sqrts_u = (
            tf.transpose(q_sqrt_u)[:, :, None]
            if q_sqrt_u.shape.ndims == 2
            else q_sqrt_u[:, None, :, :]
        )
        q_sqrts_v = (
            tf.transpose(q_sqrt_v)[:, :, None]
            if q_sqrt_v.shape.ndims == 2
            else q_sqrt_v[:, None, :, :]
        )

        base_conditional_args_to_map: Tuple[tf.Tensor, ...] = (
            Kmms,
            Kmns,
            Knns,
            Cmms,
            Cmns,
            Cnns,
            fs_u,
            fs_v,
            q_sqrts_u,
            q_sqrts_v,
            Lms,
        )

        def single_orthogonal_gp_conditional(
            t: Tuple[tf.Tensor, ...]
        ) -> MeanAndVariance:  # pragma: no cover - tf.map_fn is invisible to codecov
            Kmm, Kmn, Knn, Cmm, Cmn, Cnn, f_u, f_v, q_sqrt_u, q_sqrt_v, Lm = t

            return base_orthogonal_conditional(
                Kmn,
                Kmm,
                Knn,
                Cmn,
                Cmm,
                Cnn,
                f_u,
                f_v,
                full_cov=full_cov,
                q_sqrt_u=q_sqrt_u,
                q_sqrt_v=q_sqrt_v,
                white=white,
                Lm=Lm,
            )

    else:
        base_conditional_args_to_map = (Kmms, Kmns, Knns, Cmms, Cmns, Cnns, fs_u, fs_v, Lms)

        def single_orthogonal_gp_conditional(
            t: Tuple[tf.Tensor, ...]
        ) -> MeanAndVariance:  # pragma: no cover - tf.map_fn is invisible to codecov
            Kmm, Kmn, Knn, Cmm, Cmn, Cnn, f_u, f_v, Lm = t
            return base_orthogonal_conditional(
                Kmn,
                Kmm,
                Knn,
                Cmn,
                Cmm,
                Cnn,
                f_u,
                f_v,
                full_cov=full_cov,
                q_sqrt_u=q_sqrt_u,
                q_sqrt_v=q_sqrt_v,
                white=white,
                Lm=Lm,
            )

    rmu, rvar = tf.map_fn(
        single_orthogonal_gp_conditional,
        base_conditional_args_to_map,
        (default_float(), default_float()),
    )  # [P, N, 1], [P, 1, N, N] or [P, N, 1]

    fmu = rollaxis_left(tf.squeeze(rmu, axis=-1), 1)  # [N, P]

    if full_cov:
        fvar = tf.squeeze(rvar, axis=-3)  # [..., 0, :, :]  # [P, N, N]
    else:
        fvar = rollaxis_left(tf.squeeze(rvar, axis=-1), 1)  # [N, P]

    return fmu, fvar
