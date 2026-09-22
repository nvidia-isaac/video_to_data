# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flax actor-critic model for continuous residual actions."""

from __future__ import annotations

import math

import distrax
import flax.linen as nn
import jax.numpy as jnp

from flash_chord.training.ppo.config import NetworkConfig


class ActorCritic(nn.Module):
    """Separate actor/critic trunks with a state-independent diagonal Gaussian."""

    action_dim: int
    config: NetworkConfig

    @nn.compact
    def __call__(self, observation):
        actor = observation
        for width in self.config.actor_hidden_dims:
            actor = nn.elu(
                nn.Dense(
                    width,
                    kernel_init=nn.initializers.orthogonal(math.sqrt(2.0)),
                )(actor)
            )
        actor_output_init = nn.initializers.constant(0.0)
        if self.config.actor_output_gain > 0.0:
            actor_output_init = nn.initializers.orthogonal(self.config.actor_output_gain)
        action_mean = nn.Dense(
            self.action_dim,
            kernel_init=actor_output_init,
        )(actor)
        log_std = self.param(
            "log_std",
            nn.initializers.constant(math.log(self.config.initial_noise_std)),
            (self.action_dim,),
        )
        policy = distrax.MultivariateNormalDiag(action_mean, jnp.exp(log_std))

        critic = observation
        for width in self.config.critic_hidden_dims:
            critic = nn.elu(
                nn.Dense(
                    width,
                    kernel_init=nn.initializers.orthogonal(math.sqrt(2.0)),
                )(critic)
            )
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
        )(critic)[..., 0]
        return policy, value
