# Copyright (C) PROWLER.io 2020 - All Rights Reserved
# Unauthorised copying of this file, via any medium is strictly prohibited
# Proprietary and confidential

from typing import List, Sequence, Tuple, Optional

import tensorflow as tf
from gpflow import Parameter
from gpflow.base import _to_constrained
from gpflow.optimizers.natgrad import (
    LossClosure,
    NatGradParameters,
    NaturalGradient,
    meanvarsqrt_to_expectation,
    expectation_to_meanvarsqrt,
    meanvarsqrt_to_natural,
    XiNat,
    XiTransform,
)


class MomentumNaturalGradient(NaturalGradient):
    """
    Natural gradient optimizer with Adam-style momentum.
    Developed by Stefanos Eleftheriadis, not yet published.
    """

    def __init__(
        self,
        gamma=0.1,
        beta1=0.9,
        beta2=0.99,
        momentum=True,
        nesterov=True,
        epsilon=1e-08,
        **kwargs
    ):

        super().__init__(gamma=gamma, **kwargs)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._momentum = momentum
        self._nesterov = nesterov
        self._ms: List[tf.Variable] = None  # type: ignore
        self._v = None
        self._step_counter = None
        self.effective_lr = tf.Variable(gamma, trainable=False)

    # NOTE(Ti) we may have to include this (and test it) for saving/loading
    # Keras models compiled with this optimizer.
    #
    # def _create_slots(self, var_list):
    #     for var in var_list:
    #         self.add_slot(var, 'ms')

    def minimize(self, loss_fn: LossClosure, var_list: Sequence[NatGradParameters]):
        """
        Minimizes objective function of the model.
        Natural Gradient optimizer works with variational parameters only.
        There are two supported ways of transformation for parameters:
            - XiNat
            - XiSqrtMeanVar
        Custom transformations are also possible, they should implement
        `XiTransform` interface.

            :param loss_fn: Loss function.
            :param var_list: List of pair tuples of variational parameters or
                triplet tuple with variational parameters and ξ transformation.
                By default, all parameters goes through XiNat() transformation.
                For example your `var_list` can look as,
                ```
                var_list = [
                    (q_mu1, q_sqrt1),
                    (q_mu2, q_sqrt2, XiSqrtMeanVar())
                ]
                ```
        """
        assert len(var_list) == 1
        if self._ms is None:
            self._setup(var_list[0][:2])
        super().minimize(loss_fn, var_list)

    def _setup(self, natgrad_vars: Tuple[Parameter, Parameter]):
        self._step_counter = tf.Variable(1, dtype=tf.float64, trainable=False)
        self._ms = [
            tf.Variable(tf.zeros(tf.shape(x), dtype=tf.float64), trainable=False)
            for x in natgrad_vars
        ]
        self._v = tf.Variable(0.0, dtype=tf.float64, trainable=False)

    def _natgrad_apply_gradients(
        self,
        q_mu_grad: tf.Tensor,
        q_sqrt_grad: tf.Tensor,
        q_mu: Parameter,
        q_sqrt: Parameter,
        xi_transform: Optional[XiTransform] = None,
    ):
        if xi_transform is None:
            xi_transform = self.xi_transform

        if self._ms is None:
            self._setup((q_mu, q_sqrt))

        # 1) the ordinary gpflow gradient
        dL_dmean = _to_constrained(q_mu_grad, q_mu.transform)
        dL_dvarsqrt = _to_constrained(q_sqrt_grad, q_sqrt.transform)

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch([q_mu.unconstrained_variable, q_sqrt.unconstrained_variable])

            # the three parameterizations as functions of [q_mu, q_sqrt]
            eta1, eta2 = meanvarsqrt_to_expectation(q_mu, q_sqrt)
            # we need these to calculate the relevant gradients
            meanvarsqrt = expectation_to_meanvarsqrt(eta1, eta2)

            xi1, xi2 = xi_transform.meanvarsqrt_to_xi(q_mu, q_sqrt)
            if self._momentum:
                # Get ∂L/∂ξ via the chain rule. We need that to compute
                # the norm of the natural gradient
                _meanvarsqrt_2 = xi_transform.xi_to_meanvarsqrt(xi1, xi2)

            if not isinstance(xi_transform, XiNat):
                nat1, nat2 = meanvarsqrt_to_natural(q_mu, q_sqrt)
                xi1_nat, xi2_nat = xi_transform.naturals_to_xi(nat1, nat2)
                dummy_tensors = tf.ones_like(xi1_nat), tf.ones_like(xi2_nat)
                with tf.GradientTape(watch_accessed_variables=False) as forward_tape:
                    forward_tape.watch(dummy_tensors)
                    dummy_gradients = tape.gradient(
                        [xi1_nat, xi2_nat], [nat1, nat2], output_gradients=dummy_tensors
                    )

        # 2) the chain rule to get ∂L/∂η, where η (eta) are the expectation parameters
        dL_deta1, dL_deta2 = tape.gradient(
            meanvarsqrt, [eta1, eta2], output_gradients=[dL_dmean, dL_dvarsqrt]
        )

        if self._momentum:
            dL_dxis = tape.gradient(
                _meanvarsqrt_2, [xi1, xi2], output_gradients=[dL_dmean, dL_dvarsqrt]
            )

        if not isinstance(xi_transform, XiNat):
            nat_dL_xi1, nat_dL_xi2 = forward_tape.gradient(
                dummy_gradients, dummy_tensors, output_gradients=[dL_deta1, dL_deta2]
            )
        else:
            nat_dL_xi1, nat_dL_xi2 = dL_deta1, dL_deta2

        del tape  # Remove "persistent" tape

        nat_dL_xis = (nat_dL_xi1, nat_dL_xi2)

        # momentum
        if self._momentum:

            # adjust learning rate to debias momemtum...
            lr = (
                self.gamma
                * tf.sqrt(1.0 - self.beta2 ** self._step_counter)
                / (1.0 - self.beta1 ** self._step_counter)
            )

            # get ema for the natural gradients
            ms_new = [
                m * self.beta1 + (1.0 - self.beta1) * g
                for m, g in zip(self._ms, nat_dL_xis)
            ]
            if self._nesterov:
                ms_upd = [
                    m * self.beta1 + (1.0 - self.beta1) * g
                    for m, g in zip(ms_new, nat_dL_xis)
                ]
            else:
                ms_upd = ms_new

            # get ema for the norm of the natural gradients
            v_new = self._v * self.beta2 + (1.0 - self.beta2) * tf.reduce_sum(
                [tf.reduce_sum(g * gt) for g, gt in zip(nat_dL_xis, dL_dxis)]
            )

            # perform natural gradient descent on the ξ parameters
            xi1_new, xi2_new = [
                xi - lr * m / (tf.sqrt(v_new) + self.epsilon)
                for xi, m in zip([xi1, xi2], ms_upd)
            ]

            _ = [
                ema.assign(ema_new)
                for ema, ema_new in zip(self._ms + [self._v], ms_new + [v_new])
            ]
            self._step_counter.assign_add(1)

            effective_lr = lr / (tf.sqrt(v_new) + self.epsilon)
            self.effective_lr = effective_lr
        else:
            xi1_new, xi2_new = [
                xi - self.gamma * nat_dL_xi
                for xi, nat_dL_xi in zip([xi1, xi2], nat_dL_xis)
            ]

        # Transform back to the model parameters [q_μ, q_sqrt]
        mean_new, varsqrt_new = xi_transform.xi_to_meanvarsqrt(xi1_new, xi2_new)

        q_mu.assign(mean_new)
        q_sqrt.assign(varsqrt_new)
