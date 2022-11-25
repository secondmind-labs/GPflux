from typing import Any, Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.base import TensorLike, TensorType
from gpflow.inducing_variables import (
    InducingPoints, 
    SharedIndependentInducingVariables,
    SeparateIndependentInducingVariables
)

from gpflow.kernels import (MultioutputKernel, 
    SharedIndependent,
    SeparateIndependent
)

from gpflux.covariances.dispatch import Cvf
from check_shapes import check_shapes

@Cvf.register(InducingPoints, InducingPoints, MultioutputKernel, object)
@check_shapes(
    "inducing_variable_u: [M_u, D, 1]",
    "inducing_variable_v: [M_v, D, 1]",
    "Xnew: [batch..., N, D]",
    "return: [M_v, P, batch..., N, P]",
)
def Cvf_generic(
    inducing_variable_u: InducingPoints, 
    inducing_variable_v: InducingPoints, 
    kernel: MultioutputKernel, 
    Xnew: TensorType,
    *,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:

    return Cvf(
        inducing_variable_u,
        inducing_variable_v,
        kernel,
        Xnew,
        L_Kuu=L_Kuu,
    )  # [M, N]


@Cvf.register(
    SharedIndependentInducingVariables,
    SharedIndependentInducingVariables,
    SharedIndependent,
    object,
)
@check_shapes(
    "inducing_variable_u: [M_u, D, P]",
    "inducing_variable_v: [M_v, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [M_v, batch..., N]",
)
def Cvf_shared_shared(
    inducing_variable_u: SharedIndependentInducingVariables,
    inducing_variable_v: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
    *,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:

    return Cvf(
        inducing_variable_u.inducing_variable,
        inducing_variable_v.inducing_variable,
        kernel.kernel,
        Xnew,
        L_Kuu=L_Kuu,
    )  # [M_v, N]


@Cvf.register(SeparateIndependentInducingVariables, SeparateIndependentInducingVariables, SharedIndependent, object)
@check_shapes(
    "inducing_variable_u: [M_u, D, P]",
    "inducing_variable_v: [M_v, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [L, M_v, batch..., N]",
)
def Cvf_separate_shared(
    inducing_variable_u: SeparateIndependentInducingVariables,
    inducing_variable_v: SeparateIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: TensorType,
    *,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:

    return tf.stack([Cvf(
        ind_var_u,
        ind_var_v,
        kernel.kernel,
        Xnew,
        L_Kuu=l_kuu,
    ) for ind_var_u, ind_var_v, l_kuu in zip(inducing_variable_u.inducing_variable_list, inducing_variable_v.inducing_variable_list, L_Kuu)], axis = 0)


@Cvf.register(SharedIndependentInducingVariables, SharedIndependentInducingVariables, SeparateIndependent, object)
@check_shapes(
    "inducing_variable_u: [M_u, D, P]",
    "inducing_variable_v: [M_v, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [L, M_v, batch..., N]",
)
def Cvf_shared_separate(
    inducing_variable_u: SharedIndependentInducingVariables,
    inducing_variable_v: SharedIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: TensorType,
    *,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:

    return tf.stack([Cvf(
        inducing_variable_u.inducing_variable,
        inducing_variable_v.inducing_variable,
        k,
        Xnew,
        L_Kuu=l_kuu,
    ) for k, l_kuu in zip(kernel.kernels, L_Kuu)], axis = 0)



@Cvf.register(SeparateIndependentInducingVariables, SeparateIndependentInducingVariables, SeparateIndependent, object)
@check_shapes(
    "inducing_variable_u: [M_u, D, P]",
    "inducing_variable_v: [M_v, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [L, M_v, batch..., N]",
)
def Cvf_separate_separate(
    inducing_variable_u: SeparateIndependentInducingVariables,
    inducing_variable_v: SeparateIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: TensorType,
    *,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    n_iv_u = len(inducing_variable_u.inducing_variable_list)
    n_iv_v = len(inducing_variable_v.inducing_variable_list)
    n_k = len(kernel.kernels)
    assert (
        n_iv_u == n_k
    ), f"Must have same number of inducing variables and kernels. Found {n_iv_u} and {n_k}."

    assert (
        n_iv_v == n_k
    ), f"Must have same number of inducing variables and kernels. Found {n_iv_v} and {n_k}."


    return tf.stack([Cvf(
        ind_var_u,
        ind_var_v,
        k,
        Xnew,
        L_Kuu=l_kuu,
    ) for k, ind_var_u, ind_var_v, l_kuu in zip(kernel.kernels, inducing_variable_u.inducing_variable_list, inducing_variable_v.inducing_variable_list, L_Kuu)], axis = 0)


