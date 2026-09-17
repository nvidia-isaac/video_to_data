# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T record and inference tasks for the fixed-base Vega Sharpa embodiment.

These tasks are intentionally separate from the floating-hand Sharpa tasks. They share
only the generic HDF5/GR00T transport; their state and action contract is joint-space:
14 arm joints and 44 finger joints, with 58 absolute action targets.
"""

from __future__ import annotations

import isaaclab.envs.mdp as il_mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass

from robotic_grounding.assets import vega_camera_calib as camera_calib
from robotic_grounding.assets.vega_sharpa import (
    VEGA_ARM_JOINT_ORDER,
    VEGA_FINGER_JOINT_ORDER,
    VEGA_GROOT_JOINT_ORDER,
)
from robotic_grounding.tasks.scene_utils.visual_dr import (
    inject_scene_visual_dr_terms,
    make_visual_dr_terms,
)
from robotic_grounding.tasks.v2d import mdp as v2d_mdp
from robotic_grounding.tasks.v2d_whole_body.mdp import observations as wb_obs
from robotic_grounding.tasks.v2d_whole_body.mdp.observations import (
    record_observations,
)

from .vega_sharpa_manip_env_cfg import (
    VegaManipActionsCfg,
    VegaManipEventsCfg,
    VegaManipObservationsCfg,
    VegaSharpaManipEnvCfg,
)

_WRIST_BODIES = ("R_arm_l7", "L_arm_l7")
_CAMERA_TERM_TO_SENSOR = (
    ("image", "camera"),
    ("image_right_wrist", "camera_right_wrist"),
    ("image_left_wrist", "camera_left_wrist"),
)


@configclass
class VegaGr00tRecordObsCfg(ObsGroup):
    """Named camera and joint-space observations consumed by the Vega converter."""

    image = ObsTerm(
        func=v2d_mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "data_type": "rgb",
            "normalize": False,
        },
    )
    image_right_wrist = ObsTerm(
        func=v2d_mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("camera_right_wrist"),
            "data_type": "rgb",
            "normalize": False,
        },
    )
    image_left_wrist = ObsTerm(
        func=v2d_mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("camera_left_wrist"),
            "data_type": "rgb",
            "normalize": False,
        },
    )
    arm_joint_pos = ObsTerm(
        func=record_observations.joint_pos_ordered,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=list(VEGA_ARM_JOINT_ORDER), preserve_order=True
            )
        },
    )
    finger_joint_pos = ObsTerm(
        func=record_observations.joint_pos_ordered,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(VEGA_FINGER_JOINT_ORDER),
                preserve_order=True,
            )
        },
    )
    wrist_position_e = ObsTerm(
        func=record_observations.body_position_e,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=list(_WRIST_BODIES), preserve_order=True
            )
        },
    )
    wrist_orientation_e = ObsTerm(
        func=record_observations.body_orientation_w,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=list(_WRIST_BODIES), preserve_order=True
            )
        },
    )
    object_position_e = ObsTerm(
        func=wb_obs.object_position_e, params={"command_name": "motion"}
    )
    object_orientation_e = ObsTerm(
        func=wb_obs.object_wxyz_e, params={"command_name": "motion"}
    )

    def __post_init__(self) -> None:
        """Expose an uncorrupted dictionary without changing policy observations."""
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class VegaGr00tObservationsCfg(VegaManipObservationsCfg):
    """Native manipulation observations plus the non-policy record group."""

    record: VegaGr00tRecordObsCfg = VegaGr00tRecordObsCfg()


@configclass
class VegaGr00tEventsCfg(VegaManipEventsCfg):
    """Manipulation physics events plus embodiment-local visual randomization."""

    def __post_init__(self) -> None:
        """Attach generic visual terms to the one whole-body robot."""
        make_visual_dr_terms(self, robot_entities=("robot",))


def _add_calibrated_cameras(cfg: "VegaSharpaGr00tRecordEnvCfg") -> None:
    """Install the calibrated ego and wrist cameras without touching Sharpa configs."""
    width = camera_calib.CAMERA_RENDER_WIDTH
    height = camera_calib.CAMERA_RENDER_HEIGHT
    focal_length = camera_calib.pinhole_focal_length_mm()
    common_spawn = sim_utils.PinholeCameraCfg(
        focal_length=focal_length,
        clipping_range=(0.02, cfg.camera_far_clip),
    )

    ego_pos, ego_rot = camera_calib.ego_cam_offset_in_base()
    cfg.scene.camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root/ego_cam",
        update_period=0.0,
        width=width,
        height=height,
        data_types=["rgb"],
        spawn=common_spawn,
        offset=CameraCfg.OffsetCfg(pos=ego_pos, rot=ego_rot, convention="ros"),
    )

    wrist_common = {
        "update_period": 0.0,
        "width": width,
        "height": height,
        "data_types": ["rgb"],
        "spawn": common_spawn,
    }
    right_pos, right_rot = camera_calib.wrist_cam_offset_in_l7("right")
    left_pos, left_rot = camera_calib.wrist_cam_offset_in_l7("left")
    cfg.scene.camera_right_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/R_arm_l7/wrist_cam",
        offset=CameraCfg.OffsetCfg(pos=right_pos, rot=right_rot, convention="ros"),
        **wrist_common,
    )
    cfg.scene.camera_left_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/L_arm_l7/wrist_cam",
        offset=CameraCfg.OffsetCfg(pos=left_pos, rot=left_rot, convention="ros"),
        **wrist_common,
    )


@configclass
class VegaSharpaGr00tRecordEnvCfg(VegaSharpaManipEnvCfg):
    """Native RL-policy recording task for Dexmate/Vega Sharpa."""

    observations: VegaGr00tObservationsCfg = VegaGr00tObservationsCfg()
    events: VegaGr00tEventsCfg = VegaGr00tEventsCfg()

    camera_far_clip: float = 4.0
    gr00t_record_action_terms: tuple[str, ...] = ("joint_pos",)
    gr00t_record_action_joint_order: tuple[str, ...] = tuple(VEGA_GROOT_JOINT_ORDER)
    gr00t_record_camera_terms: tuple[tuple[str, str], ...] = _CAMERA_TERM_TO_SENSOR

    def __post_init__(self) -> None:
        """Add calibrated cameras after the native manipulation task is configured."""
        super().__post_init__()
        self.scene.env_spacing = 6.0
        self.scene.replicate_physics = False
        self.viewer.env_index = 0
        _add_calibrated_cameras(self)

    def register_scene_visual_dr_events(self) -> None:
        """Inject object and support-surface texture terms after scene resolution."""
        inject_scene_visual_dr_terms(self.events)


@configclass
class VegaGr00tActionsCfg(VegaManipActionsCfg):
    """Canonical 58-DOF absolute joint targets emitted by the GR00T profile."""

    joint_pos = il_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(VEGA_GROOT_JOINT_ORDER),
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class VegaSharpaGr00tInferenceEnvCfg(VegaSharpaGr00tRecordEnvCfg):
    """Free-running closed-loop GR00T task with the record task's exact sensors."""

    actions: VegaGr00tActionsCfg = VegaGr00tActionsCfg()

    def __post_init__(self) -> None:
        """Remove reference-only control and failure behavior from inference."""
        super().__post_init__()
        self.curriculum = None  # type: ignore[assignment]
        self.terminations.hand_wrist_away = None
        self.terminations.object_pos_error = None
        self.terminations.object_quat_error = None
        # Keep timeout and robot_state_diverged: one bounds the command trajectory and
        # the other prevents a non-finite physics state from entering the policy client.


@configclass
class VegaSharpaGr00tJointInferenceEnvCfg(VegaSharpaGr00tInferenceEnvCfg):
    """Explicit task configuration for the canonical 58-D joint contract."""
