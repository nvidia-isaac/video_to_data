# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

    from . import actions_cfg

# import logger
logger = logging.getLogger(__name__)


class JointPositionActionChunk(ActionTerm):
    """Joint action chunk term that applies the action chunk to the articulation's joints."""

    cfg: actions_cfg.JointPositionActionChunkCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(
        self, cfg: actions_cfg.JointPositionActionChunkCfg, env: ManagerBasedEnv
    ) -> None:
        """Initialize the joint position action chunk term."""
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        self._horizon = self.cfg.horizon

        # log the resolved joint names for debugging
        logger.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(
            self.num_envs, self._horizon, self.action_dim, device=self.device
        )
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._prev_targets = self._asset.data.joint_pos[:, self._joint_ids]
        self._action_chunk_execution_counter = torch.zeros(
            self.num_envs, device=self.device
        )

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.scale, self._joint_names
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict."
            )

        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.offset, self._joint_names
            )
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict."
            )

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor(
                    [[-float("inf"), float("inf")]], device=self.device
                ).repeat(self.num_envs, self.action_dim, 1)
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(
                    f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict."
                )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """The dimension of the action."""
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw actions from the policy."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions with the action chunk."""
        return self._processed_actions

    @property
    def prev_targets(self) -> torch.Tensor:
        """The previous targets of the joints."""
        return self._prev_targets

    @property
    def horizon(self) -> int:
        """The horizon of the action chunk."""
        return self._horizon

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:  # noqa: N802
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the joint action.
        It adds the following information to the base descriptor:
        - joint_names: The names of the joints.
        - scale: The scale of the action term.
        - offset: The offset of the action term.
        - clip: The clip of the action term.

        Returns:
            The IO descriptor of the action term.
        """
        super().IO_descriptor  # noqa: B018
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "JointActionChunk"
        self._IO_descriptor.joint_names = self._joint_names
        self._IO_descriptor.scale = self._scale
        # This seems to be always [4xNum_joints] IDK why. Need to check.
        if isinstance(self._offset, torch.Tensor):
            self._IO_descriptor.offset = self._offset[0].detach().cpu().numpy().tolist()
        else:
            self._IO_descriptor.offset = self._offset
        # FIXME: This is not correct. Add list support.
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

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process the actions."""
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self.prev_targets + self._raw_actions * self._scale
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        # update the previous targets
        self._prev_targets[:] = self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the joint position action chunk term."""
        self._prev_targets[env_ids] = self._asset.data.joint_pos[env_ids][
            ..., self._joint_ids
        ]
        self._raw_actions[env_ids] = 0.0

    def apply_actions(self) -> None:
        """Apply the actions."""
        # set position targets
        self._asset.set_joint_position_target(
            self.processed_actions, joint_ids=self._joint_ids
        )
