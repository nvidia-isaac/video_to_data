# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contact and wrench support rewards for whole-body manipulation.

These rewards use wrench space support functions computed by the tracking command
from retargeted contact data. The wrench space basis and friction cones are
precomputed at init; live supports are computed per step from contact sensor data.

Utility functions (compute_wrench_space, sample_wrench_space_basis_scaled, etc.)
are in tasks.v2p.mdp.utils.
"""

import torch
from isaaclab.envs import ManagerBasedEnv

from robotic_grounding.tasks.v2p.mdp.utils_jit import (
    contact_wrench_support_reward_jit,
    missed_contact_penalty_jit,
    unintended_contact_penalty_jit,
)


def contact_wrench_support_reward(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    tolerance: float = 0.1,
    var: float = 0.1,
) -> torch.Tensor:
    """Per-direction contact wrench support reward.

    Evaluates each wrench basis direction independently. Directions where both
    command and sim have support contribute exp(-loss/var), averaged over
    active command directions per body, then over bodies and hands.

    Returns continuous value in [0, 1].
    """
    command = env.command_manager.get_term(command_name)
    command.refresh_tensors()
    return contact_wrench_support_reward_jit(
        right_cmd_active=command._cached_right_wrench_cmd_active,
        right_cur_active=command._cached_right_wrench_cur_active,
        left_cmd_active=command._cached_left_wrench_cmd_active,
        left_cur_active=command._cached_left_wrench_cur_active,
        right_cmd_active_per_body=command._cached_right_wrench_cmd_active_per_body,
        left_cmd_active_per_body=command._cached_left_wrench_cmd_active_per_body,
        right_cmd_supports=command._cached_right_wrench_cmd_supports,
        right_cur_supports=command.right_contact_wrench_supports,
        left_cmd_supports=command._cached_left_wrench_cmd_supports,
        left_cur_supports=command.left_contact_wrench_supports,
        tolerance=tolerance,
        var=var,
    )


def unintended_contact_penalty(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Penalty for contact where command expects none but sim has contact.

    Combines binary indicator (unintended contact exists) with continuous
    penalty proportional to the unintended wrench support magnitude.
    """
    command = env.command_manager.get_term(command_name)
    command.refresh_tensors()
    return unintended_contact_penalty_jit(
        right_cmd_active_per_body=command._cached_right_wrench_cmd_active_per_body,
        right_cur_active_per_body=command._cached_right_wrench_cur_active_per_body,
        left_cmd_active_per_body=command._cached_left_wrench_cmd_active_per_body,
        left_cur_active_per_body=command._cached_left_wrench_cur_active_per_body,
        right_cur_supports=command.right_contact_wrench_supports,
        left_cur_supports=command.left_contact_wrench_supports,
        num_bodies=command.num_bodies,
    )


def missed_contact_penalty(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Proportional penalty for missing expected contact directions.

    For each body, computes fraction of expected wrench support directions
    missing in sim. Averages over bodies with expected contact, then hands.

    Returns continuous value in [0, 1]. Zero when no contact expected.
    """
    command = env.command_manager.get_term(command_name)
    command.refresh_tensors()
    return missed_contact_penalty_jit(
        right_cmd_active=command._cached_right_wrench_cmd_active,
        right_cur_active=command._cached_right_wrench_cur_active,
        left_cmd_active=command._cached_left_wrench_cmd_active,
        left_cur_active=command._cached_left_wrench_cur_active,
        right_cmd_active_per_body=command._cached_right_wrench_cmd_active_per_body,
        left_cmd_active_per_body=command._cached_left_wrench_cmd_active_per_body,
    )


def force_closure_reward(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    min_support: float = 0.01,
) -> torch.Tensor:
    """Force closure reward gated by binary contact labels.

    When contact is expected, rewards fraction of wrench basis directions
    with sim support above min_support. Averaged over active hands.
    This is a proxy lower bound: support is evaluated per hand rather than by
    combining both hands' contacts into a single wrench space per body.
    """
    command = env.command_manager.get_term(command_name)

    def _hand_closure(
        contact_active: torch.Tensor, cur_supports: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reward (E,), is_active (E,))."""
        # cur_supports: (E, B, M) — wrench support per body per basis direction
        # Collapse bodies: max over bodies gives best support per direction
        cur_max = cur_supports.amax(dim=1)  # (E, M)
        has_support = cur_max > min_support  # (E, M)
        fraction = has_support.float().mean(dim=-1)  # (E,) in [0, 1]

        # Gate by binary label
        is_active = contact_active > 0.5  # (E,)
        return fraction * is_active.float(), is_active

    left_reward, left_active = _hand_closure(
        command.left_hand_contact_active_command,
        command.left_hand_contact_wrench_supports,
    )
    right_reward, right_active = _hand_closure(
        command.right_hand_contact_active_command,
        command.right_hand_contact_wrench_supports,
    )

    n_hands = (left_active.float() + right_active.float()).clamp(min=1)

    return (left_reward + right_reward) / n_hands
