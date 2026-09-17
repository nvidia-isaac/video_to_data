# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorder-term and recorder-manager configs for the Sharpa V2D recording task."""
from __future__ import annotations

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from . import recorders


@configclass
class CameraRecorderCfg(RecorderTermCfg):
    """Records camera image outputs under ``camera/<sensor_name>/<data_type>``."""

    class_type: type[RecorderTerm] = recorders.CameraRecorder
    sensor_names: list[str] = ["front_cam"]
    """Names of the camera sensors in the scene to record from."""
    data_types: list[str] = [
        "rgb",
        "distance_to_image_plane",
        "instance_id_segmentation_fast",
    ]
    """Camera ``data.output`` keys to record (set by the env cfg to match the camera)."""


@configclass
class CameraInfoRecorderCfg(RecorderTermCfg):
    """Records a camera's intrinsics + extrinsics under ``camera_info/<sensor_name>``."""

    class_type: type[RecorderTerm] = recorders.CameraInfoRecorder
    sensor_names: list[str] = ["front_cam"]
    """Names of the camera sensors in the scene to record intrinsics/extrinsics from."""


@configclass
class RewardRecorderCfg(RecorderTermCfg):
    """Records the per-step reward buffer."""

    class_type: type[RecorderTerm] = recorders.RewardRecorder


@configclass
class V2DSDGRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorders (inherited) + camera images + rewards.

    Inherited terms (see ``ActionStateRecorderManagerCfg``):
      * ``record_initial_state``                    -> initial_state
      * ``record_post_step_states``                 -> states (joint pos/vel, object poses)
      * ``record_pre_step_actions``                 -> actions
      * ``record_pre_step_flat_policy_observations``-> obs (flat policy vector)
      * ``record_post_step_processed_actions``      -> processed_actions
    """

    record_camera = CameraRecorderCfg()
    record_camera_info = CameraInfoRecorderCfg()
    record_rewards = RewardRecorderCfg()
