# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Custom Isaac Lab recorder terms for camera images and rewards.

The built-in ``ActionStateRecorderManagerCfg`` already records actions, processed
actions, post-step states (joint pos/vel + object poses) and the flat policy
observation. These terms add the two things it does not: camera sensor outputs
and the per-step reward.

Each ``record_post_step`` returns ``(key, value)``. The recorder manager stores
``value[env_id]`` per episode, so the leading dim of every returned tensor must
be ``num_envs``. A returned ``dict`` is stored hierarchically as ``key/sub_key``.
"""
from __future__ import annotations

import torch
from isaaclab.managers import RecorderTerm


class CameraRecorder(RecorderTerm):
    """Records the configured image outputs of one or more camera sensors after each step.

    Reads ``cfg.sensor_names`` and ``cfg.data_types`` (see
    :class:`recorders_cfg.CameraRecorderCfg`) and emits one nested entry per
    sensor/data-type under ``camera/<sensor_name>/<data_type>``. The RecorderManager
    flattens nested dicts recursively, so ``{sensor: {data_type: tensor}}`` is stored as
    ``camera/<sensor>/<data_type>``.

    Note: a dedicated ``camera/`` top-level group is used rather than ``obs/`` to
    avoid colliding with the inherited ``PreStepFlatPolicyObservationsRecorder``,
    which writes the flat policy vector as a leaf tensor under the key ``obs``.
    """

    def record_post_step(self) -> tuple[str, dict]:
        """Return ``("camera", {sensor: {data_type: tensor}})`` for all configured sensors."""
        out = {}
        for sensor_name in self.cfg.sensor_names:
            sensor = self._env.scene.sensors[sensor_name]
            out[sensor_name] = {
                data_type: sensor.data.output[data_type].clone()
                for data_type in self.cfg.data_types
            }
        return "camera", out


class CameraInfoRecorder(RecorderTerm):
    """Records a camera sensor's intrinsics + extrinsics (world pose) after each step.

    Emits a nested entry under ``camera_info/<sensor_name>`` with:
      * ``intrinsic_matrix`` -- (num_envs, 3, 3) pinhole intrinsic matrix K
      * ``pos_w``            -- (num_envs, 3) camera position in the world frame
      * ``quat_w_world``     -- (num_envs, 4) camera orientation (w, x, y, z), world convention

    For a static camera these repeat each step; recording per-step keeps the leading dim
    equal to ``num_envs`` (as the RecorderManager requires) and stays correct once the
    camera is moved/randomized per episode.
    """

    def record_post_step(self) -> tuple[str, dict]:
        """Return ``("camera_info", {sensor: {intrinsic_matrix, pos_w, quat_w_world}})``."""
        out = {}
        for sensor_name in self.cfg.sensor_names:
            sensor = self._env.scene.sensors[sensor_name]
            out[sensor_name] = {
                "intrinsic_matrix": sensor.data.intrinsic_matrices.clone(),
                "pos_w": sensor.data.pos_w.clone(),
                "quat_w_world": sensor.data.quat_w_world.clone(),
            }
        return "camera_info", out


class RewardRecorder(RecorderTerm):
    """Records the per-step reward buffer (shape ``(num_envs,)``)."""

    def record_post_step(self) -> tuple[str, torch.Tensor]:
        """Return ``("rewards", reward_buf)`` (shape ``(num_envs,)``)."""
        return "rewards", self._env.reward_buf.clone()
