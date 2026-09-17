# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T record and closed-loop inference envs for the floating-hand Sharpa.

Two tasks, both carrying the same three-camera rig and the same ``record`` observation
group so a policy trained on recorded data sees at inference exactly what it saw in
training:

- ``Sharpa-V2D-Gr00t-Record-v0`` — an RL policy drives the hands (residual action) and
  the three camera streams plus proprio are captured for VLA post-training.
- ``Sharpa-V2D-Gr00t-Inference-v0`` — an external VLA drives the hands with ABSOLUTE
  wrist-pose + finger targets.

Both inherit visual DR from :class:`SharpaV2DDREnvCfg`. They deliberately do **not**
inherit the DR *record* env: that one installs cubicle walls and main's
``front_cam``/``ego_cam`` and relies on walls for per-env isolation, whereas this rig
isolates with a large ``env_spacing`` plus a short far clip. Mixing the two would give
four cameras and two competing isolation strategies.

Observation term names are ``image`` / ``image_right_wrist`` / ``image_left_wrist``,
which is what ``groot_finetune/convert_to_gr00t.py`` reads to build the ``front``,
``right_wrist_view`` and ``left_wrist_view`` video modalities.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.utils import configclass

from robotic_grounding.tasks.v2d import mdp
from robotic_grounding.tasks.v2d.v2d_hand_env_cfg import ObservationsCfg

from .sharpa_v2d_dr_env_cfg import SharpaV2DDREnvCfg

_TRACKING_COMMAND = "dual_hands_object_tracking_command"

# Body-attached TiledCamera has no precedent in this IsaacLab version -- the official
# wrist-cam envs (e.g. Franka stack_ik_rel_visuomotor) all use plain CameraCfg. Start
# with CameraCfg; switch only if a smoke test proves TiledCamera batches correctly under
# {ENV_REGEX_NS} when parented to a body link.
_WRIST_CAMERA_CLS = CameraCfg


def _look_at_quat_ros(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) orienting a ROS-convention camera at eye to look at target.

    ROS optical frame: +Z forward (toward target), +X right, +Y down. Frame-agnostic --
    passing link-frame eye/target/up yields the link-frame quaternion, which is what the
    wrist mounts below rely on.
    """
    e = np.asarray(eye, dtype=float)
    t = np.asarray(target, dtype=float)
    u = np.asarray(up, dtype=float)
    z = t - e
    z /= np.linalg.norm(z)
    x = np.cross(z, u)
    if np.linalg.norm(x) < 1e-6:  # looking straight up/down; pick another up
        x = np.cross(z, np.array([0.0, 1.0, 0.0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.stack([x, y, z], axis=1)  # columns are the camera axes in world
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    return (float(w), float(qx), float(qy), float(qz))


#################################################
# Observations
#################################################


@configclass
class Gr00tRecordObsCfg(ObsGroup):
    """Non-policy group captured for VLA training: three views plus proprio.

    ``concatenate_terms=False`` so the group surfaces as a dict of named terms, which the
    re-render driver writes verbatim as ``data/demo_i/obs/<term>``. Does not affect the
    policy or critic.
    """

    image = ObsTerm(
        func=mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("camera"),
            "data_type": "rgb",
            "normalize": False,
        },
    )
    image_right_wrist = ObsTerm(
        func=mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("camera_right_wrist"),
            "data_type": "rgb",
            "normalize": False,
        },
    )
    image_left_wrist = ObsTerm(
        func=mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("camera_left_wrist"),
            "data_type": "rgb",
            "normalize": False,
        },
    )
    wrist_position_e = ObsTerm(
        func=mdp.wrist_position_e, params={"command_name": _TRACKING_COMMAND}
    )
    wrist_orientation_e = ObsTerm(
        func=mdp.wrist_orientation_e, params={"command_name": _TRACKING_COMMAND}
    )
    finger_joint_pos = ObsTerm(
        func=mdp.finger_joint_pos, params={"command_name": _TRACKING_COMMAND}
    )
    object_position_e = ObsTerm(
        func=mdp.object_position_e, params={"command_name": _TRACKING_COMMAND}
    )
    object_orientation_e = ObsTerm(
        func=mdp.object_orientation_e, params={"command_name": _TRACKING_COMMAND}
    )
    processed_right_action = ObsTerm(
        func=mdp.processed_action,
        params={"action_name": "right_joint_residual_action"},
    )
    processed_left_action = ObsTerm(
        func=mdp.processed_action,
        params={"action_name": "left_joint_residual_action"},
    )

    def __post_init__(self) -> None:
        """Surface as named terms, uncorrupted."""
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class Gr00tObservationsCfg(ObservationsCfg):
    """Base observations plus the ``record`` group (policy group unchanged)."""

    record: Gr00tRecordObsCfg = Gr00tRecordObsCfg()


#################################################
# Camera rig
#################################################

# Third-person view. Tight on the env's OWN robot: a VLA must not see neighbouring envs
# or their shadows. Isolation comes from a large env_spacing plus a far clip shorter than
# that spacing, so neighbours are simply not rendered.
_CAMERA_EYE = (0.0, -0.7, 1.5)
_CAMERA_TARGET = (0.0, -0.05, 1.03)

# MOCKED wrist-camera mounts -- NOT a calibrated rig.
#
# These poses are a plausible stand-in for the real robot's wrist cameras, hand-picked in
# the viewer, not measured. They are close enough to produce sensible near-field hand and
# object framing for pipeline development, but any policy trained on these images will not
# transfer to the physical robot without re-baking against the real mount geometry.
#
# Frame convention (wrist / C_MC link): +Z distal (toward the fingers), right hand -Y and
# left hand +Y = ulnar (pinky) side, +-X = palm normal.
_WRIST_CAM_WIDTH: int = 224
_WRIST_CAM_HEIGHT: int = 224
# ~100 deg horizontal FOV for wide near-field framing; the third-person cam is ~53 deg.
_WRIST_CAM_FOCAL_LENGTH: float = 9.0
_RIGHT_WRIST_CAM_POS: tuple = (0.0, -0.085, -0.05)
_RIGHT_WRIST_CAM_TARGET: tuple = (0.0, -0.01, 0.2)
_RIGHT_WRIST_CAM_UP: tuple = (1.0, 0.0, 0.0)
_LEFT_WRIST_CAM_POS: tuple = (0.0, 0.085, -0.05)
_LEFT_WRIST_CAM_TARGET: tuple = (0.0, 0.01, 0.2)
_LEFT_WRIST_CAM_UP: tuple = (1.0, 0.0, 0.0)


def _add_cameras(cfg: "SharpaV2DGr00tRecordEnvCfg") -> None:
    """Attach the third-person camera and the two wrist cameras to ``cfg.scene``.

    The wrist cameras are parented under the hand's C_MC link, so their ``OffsetCfg`` is
    interpreted in that link's frame and they track the hand rigidly.
    """
    cfg.scene.env_spacing = cfg.env_spacing
    cfg.scene.camera = TiledCameraCfg(
        prim_path=cfg.camera_prim_path,
        update_period=0.0,
        width=cfg.camera_width,
        height=cfg.camera_height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cfg.camera_focal_length,
            clipping_range=(0.05, cfg.camera_far_clip),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=cfg.camera_eye,
            rot=_look_at_quat_ros(cfg.camera_eye, cfg.camera_target),
            convention="ros",
        ),
    )

    wrist_common = dict(
        update_period=0.0,
        width=cfg.wrist_camera_width,
        height=cfg.wrist_camera_height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cfg.wrist_camera_focal_length,
            clipping_range=(0.02, cfg.camera_far_clip),
        ),
    )
    cfg.scene.camera_right_wrist = _WRIST_CAMERA_CLS(
        prim_path="{ENV_REGEX_NS}/RightRobot/right_hand_C_MC/wrist_cam",
        offset=CameraCfg.OffsetCfg(
            pos=cfg.right_wrist_cam_pos,
            rot=_look_at_quat_ros(
                cfg.right_wrist_cam_pos,
                cfg.right_wrist_cam_target,
                up=cfg.right_wrist_cam_up,
            ),
            convention="ros",
        ),
        **wrist_common,
    )
    cfg.scene.camera_left_wrist = _WRIST_CAMERA_CLS(
        prim_path="{ENV_REGEX_NS}/LeftRobot/left_hand_C_MC/wrist_cam",
        offset=CameraCfg.OffsetCfg(
            pos=cfg.left_wrist_cam_pos,
            rot=_look_at_quat_ros(
                cfg.left_wrist_cam_pos,
                cfg.left_wrist_cam_target,
                up=cfg.left_wrist_cam_up,
            ),
            convention="ros",
        ),
        **wrist_common,
    )


#################################################
# Env cfgs
#################################################


@configclass
class SharpaV2DGr00tRecordEnvCfg(SharpaV2DDREnvCfg):
    """Visual-DR manipulation env with a three-camera ``record`` observation group."""

    gr00t_record_action_terms: tuple[str, ...] = (
        "right_joint_residual_action",
        "left_joint_residual_action",
    )
    gr00t_record_action_joint_order: tuple[str, ...] | None = None
    gr00t_record_camera_terms: tuple[tuple[str, str], ...] = (
        ("image", "camera"),
        ("image_right_wrist", "camera_right_wrist"),
        ("image_left_wrist", "camera_left_wrist"),
    )

    # Third-person camera (env frame, look-at).
    camera_prim_path: str = "{ENV_REGEX_NS}/Camera"
    camera_eye: tuple = _CAMERA_EYE
    camera_target: tuple = _CAMERA_TARGET
    camera_width: int = 224
    camera_height: int = 224
    camera_focal_length: float = 18.0
    # Neighbours sit at env_spacing; a far clip below that culls them and their shadows.
    env_spacing: float = 6.0
    camera_far_clip: float = 4.0

    # Mocked wrist cameras -- see the mount note above.
    wrist_camera_width: int = _WRIST_CAM_WIDTH
    wrist_camera_height: int = _WRIST_CAM_HEIGHT
    wrist_camera_focal_length: float = _WRIST_CAM_FOCAL_LENGTH
    right_wrist_cam_pos: tuple = _RIGHT_WRIST_CAM_POS
    right_wrist_cam_target: tuple = _RIGHT_WRIST_CAM_TARGET
    right_wrist_cam_up: tuple = _RIGHT_WRIST_CAM_UP
    left_wrist_cam_pos: tuple = _LEFT_WRIST_CAM_POS
    left_wrist_cam_target: tuple = _LEFT_WRIST_CAM_TARGET
    left_wrist_cam_up: tuple = _LEFT_WRIST_CAM_UP

    def __post_init__(self) -> None:
        """Install the camera rig and the record observation group."""
        super().__post_init__()
        _add_cameras(self)
        self.observations = Gr00tObservationsCfg()
        # V2DHandEnvCfg pins the viewer to env 6, which raises at construction for any
        # --num_envs <= 6. A data-generation env has to work at small env counts, so pin
        # to env 0 (what the base recording cfg does too).
        self.viewer.env_index = 0


@configclass
class SharpaV2DGr00tInferenceEnvCfg(SharpaV2DGr00tRecordEnvCfg):
    """Closed-loop inference env: an external VLA drives the hands with ABSOLUTE targets.

    Inherits the camera rig, ``record`` group and visual DR from the record env, so the
    VLA sees the same images and proprio it was trained on. Differs by:

    - **Action terms swapped in place**, keeping the attribute names
      ``right/left_joint_residual_action``, so the inherited ``processed_action`` obs terms
      and the ``action_l1`` reward still resolve -- renaming would ``KeyError`` at env
      construction. Declaration order stays ``[right, left]``, so the concatenated
      ``29 + 29`` layout matches the GR00T action-key order.
    - **Trajectory-deviation terminations dropped**, since the policy no longer follows the
      reference. ``time_out`` is kept: it bounds the episode and keeps the command's
      ``timestep_counter`` in range. The command term still loads (it seeds the object and
      scene at reset) but no longer drives the hands.

    Swapping residual (28 raw per hand) for absolute (29 per hand) changes the policy
    group's ``last_action`` obs from 56 to 58. Harmless here because no policy network
    consumes it, but do not reuse this env with a residual-trained RL checkpoint.
    """

    def __post_init__(self) -> None:
        """Swap to absolute-pose actions and drop reference-tracking terminations."""
        super().__post_init__()

        # Reuse the residual terms' wrist-controller gains (see v2d_hand_env_cfg.py).
        self.actions.right_joint_residual_action = mdp.JointAbsolutePoseActionCfg(
            asset_name="right_robot",
            joint_names=[".*"],
            tracking_controller_linear_stiffness=50.0,
            tracking_controller_linear_damping=10.0,
            tracking_controller_angular_stiffness=12.0,
            tracking_controller_angular_damping=0.1,
        )
        self.actions.left_joint_residual_action = mdp.JointAbsolutePoseActionCfg(
            asset_name="left_robot",
            joint_names=[".*"],
            tracking_controller_linear_stiffness=50.0,
            tracking_controller_linear_damping=10.0,
            tracking_controller_angular_stiffness=12.0,
            tracking_controller_angular_damping=0.1,
        )

        self.terminations.hand_wrist_away_from_trajectory = None
        self.terminations.object_away_from_trajectory = None
