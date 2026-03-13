# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Shared utilities for converting hand-object datasets to ManoSharpaData schema.

Use this module from dataset-specific conversion scripts (e.g. ARCTIC, TACO)
to avoid duplicating IK setup, per-frame IK, and saving logic.
"""

from typing import Any, Literal, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from robotic_grounding.retarget import SHARPA_WAVE_XMLS_DIR
from robotic_grounding.retarget.hand_kinematics import (
    HandKinematics,
    SharpaHandKinematics,
)
from robotic_grounding.retarget.params import MANO_FINGERTIP_INDICES

# Default partition columns when saving to Parquet (shared across datasets)
DEFAULT_PARTITION_COLS = ["sequence_id", "robot_name"]


def setup_sharpa_kinematics(
    side: Literal["right", "left"],
    frequency: float = 200.0,
    frame_tasks_converged_threshold: float = 1e-6,
) -> HandKinematics:
    """Create HandKinematics for the Sharpa hand.

    Args:
        side: "right" or "left".
        frequency: Solver frequency (Hz).
        frame_tasks_converged_threshold: Convergence threshold for IK.

    Returns:
        HandKinematics instance for the given side.
    """
    robot_asset_path = str(SHARPA_WAVE_XMLS_DIR / f"{side}_sharpawave.xml")
    return SharpaHandKinematics(
        side=side,
        robot_asset_path=robot_asset_path,
        source_model="mano",
        use_relative_frames=False,
        frequency=frequency,
        frame_tasks_converged_threshold=frame_tasks_converged_threshold,
    )


def run_frame_ik(
    right_kinematics: HandKinematics,
    left_kinematics: HandKinematics,
    right_mano_joints: torch.Tensor,
    right_mano_joints_wxyz: torch.Tensor,
    left_mano_joints: torch.Tensor,
    left_mano_joints_wxyz: torch.Tensor,
    mano_to_robot_scale: float,
    right_qpos_prev: Optional[np.ndarray] = None,
    left_qpos_prev: Optional[np.ndarray] = None,
    right_wrist_position: Optional[np.ndarray] = None,
    right_wrist_quat_xyzw: Optional[np.ndarray] = None,
    left_wrist_position: Optional[np.ndarray] = None,
    left_wrist_quat_xyzw: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Run IK for one frame for both hands.

    When right_qpos_prev (or left_qpos_prev) is None, it is initialized from
    right_wrist_position and right_wrist_quat_xyzw (and left equivalents).
    Quat must be in Pinocchio/qpos order (xyzw). After the first frame, pass
    the returned right_qpos/left_qpos as right_qpos_prev/left_qpos_prev.

    Args:
        right_kinematics: Right-hand kinematics.
        left_kinematics: Left-hand kinematics.
        right_mano_joints: Right MANO joints (21, 3).
        right_mano_joints_wxyz: Right MANO joint quats wxyz (21, 4).
        left_mano_joints: Left MANO joints (21, 3).
        left_mano_joints_wxyz: Left MANO joint quats wxyz (21, 4).
        mano_to_robot_scale: Scale from MANO to robot.
        right_qpos_prev: Previous right qpos (None to initialize from wrist).
        left_qpos_prev: Previous left qpos (None to initialize from wrist).
        right_wrist_position: Used when right_qpos_prev is None (3,).
        right_wrist_quat_xyzw: Used when right_qpos_prev is None (4,) xyzw.
        left_wrist_position: Used when left_qpos_prev is None (3,).
        left_wrist_quat_xyzw: Used when left_qpos_prev is None (4,) xyzw.

    Returns:
        (right_qpos, left_qpos, right_kinematics_results, left_kinematics_results).
    """
    if right_qpos_prev is None:
        right_qpos = right_kinematics.robot.q0.copy()
        right_qpos[:3] = right_wrist_position
        right_qpos[3:7] = right_wrist_quat_xyzw
    else:
        right_qpos = right_qpos_prev.copy()

    if left_qpos_prev is None:
        left_qpos = left_kinematics.robot.q0.copy()
        left_qpos[:3] = left_wrist_position
        left_qpos[3:7] = left_wrist_quat_xyzw
    else:
        left_qpos = left_qpos_prev.copy()

    right_results = right_kinematics.compute(
        right_mano_joints,
        right_mano_joints_wxyz,
        source_to_robot_scale=mano_to_robot_scale,
        qpos=right_qpos,
    )
    right_qpos = right_results["q"]

    left_results = left_kinematics.compute(
        left_mano_joints,
        left_mano_joints_wxyz,
        source_to_robot_scale=mano_to_robot_scale,
        qpos=left_qpos,
    )
    left_qpos = left_results["q"]

    return right_qpos, left_qpos, right_results, left_results


def compute_tip_to_object_surface_distance(
    mano_joints: torch.Tensor,
    object_surface_points_world: torch.Tensor,
) -> Optional[list[list[float]]]:
    """Compute MANO fingertip-to-surface distances for one hand.

    Object surface points are assumed to be in world frame.

    Args:
        mano_joints: MANO joints (21, 3) on same device as object_surface_points_world.
        object_surface_points_world: Object surface points in world frame (V, 3).

    Returns:
        List of 5 lists (one per fingertip) of float distances, or None if
        computation is not possible.
    """
    fingertips = mano_joints[MANO_FINGERTIP_INDICES]

    dists = torch.cdist(
        fingertips.unsqueeze(0), object_surface_points_world.unsqueeze(0)
    ).squeeze(0)

    # Get minimum distance from each fingertip to surface
    min_dists = dists.amin(dim=-1)  # (5,)

    return min_dists.cpu().tolist()


def wrist_pose_from_mano_joint0(
    joint0_position: np.ndarray,
    joint0_wxyz: np.ndarray,
    link_to_site_quat_xyzw: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute robot wrist position and quat (xyzw) from MANO wrist (joint 0).

    Optionally applies a rotation offset (e.g. from link frame to site).
    Default offset used in ARCTIC conversion: [0.5, -0.5, 0.5, 0.5] in wxyz.

    Args:
        joint0_position: MANO wrist position (3,).
        joint0_wxyz: MANO wrist quaternion wxyz (4,).
        link_to_site_quat_xyzw: If given, rotation offset in xyzw applied
            to the right (e.g. link to site). Default None = no offset.

    Returns:
        (position, quat_xyzw) for use as wrist_position and wrist_quat_xyzw
        in run_frame_ik().
    """
    pos = np.asarray(joint0_position, dtype=np.float64)
    q_mano = R.from_quat(
        np.asarray(joint0_wxyz, dtype=np.float64).reshape(4),
        scalar_first=True,
    )
    if link_to_site_quat_xyzw is not None:
        q_offset = R.from_quat(
            np.asarray(link_to_site_quat_xyzw, dtype=np.float64).reshape(4),
            scalar_first=False,
        )
        q_mano = q_mano * q_offset.inv()
    quat_xyzw = q_mano.as_quat(scalar_first=False)
    return pos, quat_xyzw
