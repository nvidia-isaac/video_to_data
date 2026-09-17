# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Residual joint-position action around a time-varying motion reference.

The action target is ``reference_joint_pos[t] + filtered_residual``. A zero policy output
therefore replays the reference instead of holding the robot at its current position.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from .reference_residual_action_cfg import ReferenceResidualJointPositionActionCfg

logger = logging.getLogger(__name__)


class ReferenceResidualJointPositionAction(ActionTerm):
    """``target = reference_joint_pos[t] + ema_filtered(clamp(raw * scale))``."""

    cfg: ReferenceResidualJointPositionActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(
        self, cfg: ReferenceResidualJointPositionActionCfg, env: ManagerBasedEnv
    ) -> None:
        """Resolve the acted-on joints and allocate the raw/residual buffers."""
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names,
        )
        self._num_joints = len(self._joint_ids)
        logger.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        self._raw_actions = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # Evidence of the target most recently sent to the articulation. ManagerBasedEnv
        # resets completed environments before env.step() returns and reset() correctly
        # clears _processed_actions. Rollout exporters nevertheless need the target that
        # was actually applied on that terminal step, so preserve it in a buffer whose
        # lifetime is independent of the next episode's action-manager state.
        self._last_applied_actions = torch.zeros_like(self._raw_actions)
        # EMA state on the residual.
        self._filtered_residual = torch.zeros_like(self._raw_actions)

        # Columns of command_joint_pos matching this term's joints; resolved lazily because
        # the command manager does not exist yet when action terms are constructed.
        self._ref_cols: torch.Tensor | slice | None = None

        self._scale: float | torch.Tensor
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.scale, self._joint_names
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported scale type: {type(cfg.scale)}. "
                "Supported types are float and dict."
            )

        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            elif isinstance(cfg.clip, float):
                self._clip = torch.tensor(
                    [[-cfg.clip, cfg.clip]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. "
                    "Supported types are float and dict."
                )

        logger.info(
            f"{self.__class__.__name__}: reference-anchored residual over "
            f"{self._num_joints} joints (command={self.cfg.command_name!r}, "
            f"ema={self.cfg.ema_factor}, clip_to_joint_limits="
            f"{self.cfg.clip_to_joint_limits})"
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Number of joints this term drives."""
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        """The unscaled policy output."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The composed absolute joint position targets."""
        return self._processed_actions

    @property
    def last_applied_actions(self) -> torch.Tensor:
        """Absolute targets issued on the latest physics step, including terminal steps.

        Unlike :attr:`processed_actions`, this buffer is deliberately not cleared by
        ``reset``. This lets a caller inspect the action after ``env.step`` even when
        IsaacLab auto-reset the completed environment before returning.
        """
        return self._last_applied_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:  # noqa: N802
        """Describe joint names, scale and clip for action introspection."""
        super().IO_descriptor  # noqa: B018
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "ReferenceResidualJointPositionAction"
        self._IO_descriptor.joint_names = self._joint_names
        self._IO_descriptor.scale = self._scale
        if self.cfg.clip is not None:
            if isinstance(self._clip, torch.Tensor):
                self._IO_descriptor.clip = self._clip[0].detach().cpu().numpy().tolist()
            else:
                self._IO_descriptor.clip = self._clip
        else:
            self._IO_descriptor.clip = None
        return self._IO_descriptor

    """
    Operations.
    """

    def _reference_joint_pos(self) -> torch.Tensor:
        """Reference joint targets for this term's joints. (E, num_joints).

        ``command_joint_pos`` is laid out over the command's OWN tracked joints (already
        reordered into the asset's joint order by the tracking command). Match it to this
        term's joints by NAME rather than assuming the two joint sets coincide -- they do
        today (both resolve ``".*"``) but a silent mismatch would scramble every joint
        target.
        """
        command = self._env.command_manager.get_term(self.cfg.command_name)
        if self._ref_cols is None:
            cmd_names = list(getattr(command, "_tracked_joint_names", None) or [])
            if not cmd_names or cmd_names == list(self._joint_names):
                self._ref_cols = slice(None)
            else:
                missing = [n for n in self._joint_names if n not in cmd_names]
                if missing:
                    raise ValueError(
                        f"{self.__class__.__name__}: command {self.cfg.command_name!r} "
                        f"provides no reference for joints {missing}"
                    )
                self._ref_cols = torch.tensor(
                    [cmd_names.index(n) for n in self._joint_names],
                    dtype=torch.long,
                    device=self.device,
                )
        return command.command_joint_pos[:, self._ref_cols]

    def process_actions(self, actions: torch.Tensor) -> None:
        """Compose the absolute target from the reference plus a filtered residual."""
        self._raw_actions[:] = actions
        scaled = self._raw_actions * self._scale
        # EMA first, then symmetric clip -- order matters; clipping first would let the
        # filter re-inflate the residual past the bound.
        self._filtered_residual = (
            self.cfg.ema_factor * self._filtered_residual
            + (1.0 - self.cfg.ema_factor) * scaled
        )
        if self.cfg.clip is not None:
            self._filtered_residual = self._filtered_residual.clamp(
                min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        target = self._reference_joint_pos() + self._filtered_residual
        if self.cfg.clip_to_joint_limits:
            limits = self._asset.data.joint_pos_limits[:, self._joint_ids, :]
            target = target.clamp(min=limits[..., 0], max=limits[..., 1])
        self._processed_actions = target

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear the residual so a reset env starts exactly on the reference pose."""
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
            self._filtered_residual[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._filtered_residual[env_ids] = 0.0

    def apply_actions(self) -> None:
        """Write the composed targets to the articulation's PD controller."""
        self._last_applied_actions.copy_(self._processed_actions)
        self._asset.set_joint_position_target(
            self.processed_actions, joint_ids=self._joint_ids
        )
