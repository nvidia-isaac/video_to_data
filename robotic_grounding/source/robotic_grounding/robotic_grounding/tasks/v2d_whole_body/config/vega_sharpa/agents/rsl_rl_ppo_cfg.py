# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class VegaSharpaWholeBodyRslRlPpoCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for the Vega Sharpa whole-body joint-tracking env."""

    num_steps_per_env = 24
    # Leave training time after the VOC curriculum reaches zero at iteration 12,500.
    max_iterations = 20_000
    save_interval = 500
    experiment_name = "vega_sharpa_whole_body"
    empirical_normalization = True

    # Conservative exploration prevents learned action noise from overwhelming tracking.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.1,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )
