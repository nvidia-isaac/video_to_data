"""Contact and wrench support rewards for whole-body manipulation.

These rewards use wrench space support functions computed by the tracking command
from retargeted contact data. The wrench space basis and friction cones are
precomputed at init; live supports are computed per step from contact sensor data.

Utility functions (compute_wrench_space, sample_wrench_space_basis_scaled, etc.)
are in tasks.v2p.mdp.utils.
"""

import torch
from isaaclab.envs import ManagerBasedEnv


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

    def _hand_reward(
        cmd: torch.Tensor, cur: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cmd_active = cmd > 1e-3  # (E, B, M)
        cur_active = cur > 1e-3
        n_cmd_per_body = cmd_active.sum(dim=-1).clamp(min=1)  # (E, B)

        # Tolerance band violations per direction
        lower = ((1.0 - tolerance) * cmd - cur).clamp(min=0.0)
        upper = (cur - (1.0 + tolerance) * cmd).clamp(min=0.0)
        loss = lower.square() + upper.square()  # (E, B, M)

        # Per-direction reward, gated by both command and sim having support
        per_dir_reward = (cmd_active & cur_active) * torch.exp(-loss / var)
        per_body_reward = per_dir_reward.sum(dim=-1) / n_cmd_per_body  # (E, B)

        # Average over bodies with command contact
        body_has_contact = cmd_active.any(dim=-1)  # (E, B)
        n_bodies = body_has_contact.sum(dim=-1).clamp(min=1)
        hand_reward = (per_body_reward * body_has_contact).sum(dim=-1) / n_bodies
        return hand_reward, body_has_contact.any(dim=-1)

    right_reward, right_any = _hand_reward(
        command.right_hand_contact_wrench_supports_command,
        command.right_hand_contact_wrench_supports,
    )
    left_reward, left_any = _hand_reward(
        command.left_hand_contact_wrench_supports_command,
        command.left_hand_contact_wrench_supports,
    )

    n_hands = (right_any.float() + left_any.float()).clamp(min=1)
    return (right_reward + left_reward) / n_hands


def unintended_contact_penalty(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Penalty for contact where command expects none but sim has contact.

    Combines binary indicator (unintended contact exists) with continuous
    penalty proportional to the unintended wrench support magnitude.
    """
    command = env.command_manager.get_term(command_name)

    def _hand_penalty(
        cmd_supports: torch.Tensor, cur_supports: torch.Tensor
    ) -> torch.Tensor:
        cmd_has = cmd_supports.amax(dim=-1) > 1e-3  # (E, B)
        cur_has = cur_supports.amax(dim=-1) > 1e-3
        n_cmd = cmd_has.sum(dim=-1)  # (E,)
        num_bodies = cmd_supports.shape[1]

        # Binary: contact where not expected
        unintended = (~cmd_has) & cur_has  # (E, B)
        binary = unintended.float().mean(dim=-1)

        # Continuous: wrench magnitude where not expected
        continuous = (~cmd_has).float() * cur_supports.clamp(min=0.0).square().mean(
            dim=-1
        )
        continuous = continuous.sum(dim=-1) / (num_bodies - n_cmd).clamp(min=1e-3)

        return binary + continuous

    return _hand_penalty(
        command.right_hand_contact_wrench_supports_command,
        command.right_hand_contact_wrench_supports,
    ) + _hand_penalty(
        command.left_hand_contact_wrench_supports_command,
        command.left_hand_contact_wrench_supports,
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

    def _hand_penalty(cmd: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        cmd_active = cmd > 1e-3  # (E, B, M)
        cur_active = cur > 1e-3
        missed = cmd_active & ~cur_active
        n_expected = cmd_active.sum(dim=-1)  # (E, B)
        n_missed = missed.sum(dim=-1)
        frac = n_missed / n_expected.clamp(min=1)  # (E, B)

        body_has_contact = n_expected > 0  # (E, B)
        n_bodies = body_has_contact.sum(dim=-1).clamp(min=1)
        return (frac * body_has_contact).sum(dim=-1) / n_bodies

    right_cmd = command.right_hand_contact_wrench_supports_command
    right_cur = command.right_hand_contact_wrench_supports
    left_cmd = command.left_hand_contact_wrench_supports_command
    left_cur = command.left_hand_contact_wrench_supports

    right_any = (right_cmd > 1e-3).any(dim=-1).any(dim=-1)
    left_any = (left_cmd > 1e-3).any(dim=-1).any(dim=-1)
    n_hands = (right_any.float() + left_any.float()).clamp(min=1)

    return (
        _hand_penalty(right_cmd, right_cur) + _hand_penalty(left_cmd, left_cur)
    ) / n_hands


# TODO: force closure should be explicitly evaluated (this is a proxy lower bound).
# TODO: combine both hands' contacts into a single wrench space per body for
# bimanual grasps, rather than evaluating per-hand independently.
def force_closure_reward(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    min_support: float = 0.01,
) -> torch.Tensor:
    """Force closure reward gated by binary contact labels.

    When contact is expected, rewards fraction of wrench basis directions
    with sim support above min_support. Averaged over active hands.
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
