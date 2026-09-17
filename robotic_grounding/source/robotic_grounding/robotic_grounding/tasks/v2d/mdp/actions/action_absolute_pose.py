# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Absolute wrist-pose + finger-target action term for closed-loop VLA control.

A VLA (GR00T) predicts an **absolute** per-hand target — wrist position (3) + wrist
orientation quaternion wxyz (4) + finger joint positions (N) = 7 + N — rather than a
residual on a reference trajectory. This term applies that target directly.

It reuses the Cartesian-impedance wrench + finger-PD controller of
:class:`~.action_track_residual.JointResidualWithTrackingAction` verbatim (same
``apply_actions``, which drives the wrist link via ``set_external_force_and_torque`` and
sets finger joint position targets, with torque-limit clamping). The only differences vs.
the residual term:

- ``action_dim`` is ``7 + N`` (wrist quaternion is taken directly, not an Euler delta), so
  the processed buffer width equals the action width (no residual ``+1`` euler->quat).
- ``process_actions`` writes the raw action straight into the processed buffer (unit-
  normalizing the wrist quaternion); **no** scale / clip / EMA and **no** command
  reference is added.

The term still needs ``command_name``: the base resolves the robot, wrist body id, finger
joint ids, and the *current measured* wrist pose (used by the PD controller's position
error) from the command term. Only the command's *reference target* is bypassed. The
command term must therefore still be present/loaded in the env (it also seeds the
object/scene at reset), but it no longer drives the hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from robotic_grounding.tasks.v2d.mdp.actions.action_track_residual import (
    JointResidualWithTrackingAction,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class JointAbsolutePoseAction(JointResidualWithTrackingAction):
    """Apply an absolute wrist pose (pos + wxyz quat) + finger joint targets from a VLA."""

    cfg: actions_cfg.JointAbsolutePoseActionCfg

    def __init__(
        self, cfg: actions_cfg.JointAbsolutePoseActionCfg, env: ManagerBasedEnv
    ) -> None:
        """Resize the processed-action buffer for an absolute quaternion target."""
        super().__init__(cfg, env)
        # The base sizes `_processed_actions` to `action_dim + 1` because it stores the
        # wrist orientation as a quaternion while the residual action carries Euler (3).
        # Here the action already carries the quaternion (4), so processed width ==
        # action_dim. Reallocate to `[pos(3), quat(4), finger(N)]` with an identity quat.
        n_finger = len(self.finger_joint_ids)
        self._processed_actions = torch.zeros(
            self.num_envs, 3 + 4 + n_finger, device=self.device
        )
        self._processed_actions[..., 3] = 1.0  # identity wxyz

    @property
    def action_dim(self) -> int:
        """Wrist position (3) + wrist quaternion wxyz (4) + finger joints (N)."""
        return len(self.finger_joint_ids) + 3 + 4

    def process_actions(self, actions: torch.Tensor) -> None:
        """Write the absolute target directly: ``[pos(3), quat wxyz(4), finger(N)]``.

        No scale / clip / EMA / command reference. The wrist quaternion is unit-normalized
        (the VLA may emit a non-unit quaternion). Finger targets are clamped to the joint
        torque limits downstream in the inherited ``apply_actions``.
        """
        self._raw_actions[:] = actions
        pos = actions[:, :3]
        quat = actions[:, 3:7]
        finger = actions[:, 7:]

        norm = torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
        quat = quat / norm

        self._processed_actions[:, :3] = pos
        self._processed_actions[:, 3:7] = quat
        self._processed_actions[:, 7:] = finger
