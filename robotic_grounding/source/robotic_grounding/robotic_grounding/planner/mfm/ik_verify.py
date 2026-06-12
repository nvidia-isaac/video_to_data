# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IK verification for planner EE targets using differential IK (mink).

Optional dependency: ``pip install mink`` is required to use this module.
"""

# ruff: noqa: ANN001, ANN201, ANN202, ANN204, D102, D103, D107, D417
# Planner is still in active development and this file is likely to change
# significantly with the new groot planner. Suppress annotation/docstring
# lint for now; real code issues are fixed individually.

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def run_ik_verification(
    xml_path: str,
    ref_data: dict,
    fps: float = 30.0,
    ik_steps: int = 500,
    dt: float = 0.002,
) -> dict:
    """Verify EE target reachability using differential IK.

    Args:
        xml_path: Path to MuJoCo XML.
        ref_data: Dict with left_pos, left_quat, right_pos, right_quat (wxyz).
        fps: Playback frame rate.
        ik_steps: Number of IK iterations per frame.
        dt: Integration timestep.

    Returns:
        Dict with qpos_log, pos_errors_l/r, ori_errors_l/r.

    Raises:
        ImportError: If mink is not installed.
    """
    try:
        import mink  # noqa: PLC0415
    except ImportError as err:
        raise ImportError(
            "mink is required for IK verification: pip install mink\n"
            "This is an optional debug tool, not needed for training."
        ) from err

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)

    tasks = [
        mink.FrameTask("left_wrist_yaw_link", position_cost=2.0, orientation_cost=2.0),
        mink.FrameTask("right_wrist_yaw_link", position_cost=2.0, orientation_cost=2.0),
        mink.FrameTask("pelvis", position_cost=10.0),
        mink.FrameTask("left_ankle_roll_link", position_cost=10.0),
        mink.FrameTask("right_ankle_roll_link", position_cost=10.0),
        mink.PostureTask(model, cost=0.1),
    ]

    T = len(ref_data["left_pos"])
    qpos_log = []
    pos_errors_l, pos_errors_r = [], []
    _ori_errors_l: list[float] = []
    _ori_errors_r: list[float] = []

    for frame in range(T):
        # Set targets
        left_se3 = _pose_to_se3(
            ref_data["left_pos"][frame], ref_data["left_quat"][frame]
        )
        right_se3 = _pose_to_se3(
            ref_data["right_pos"][frame], ref_data["right_quat"][frame]
        )
        tasks[0].set_target(left_se3)
        tasks[1].set_target(right_se3)

        # Solve IK
        for _ in range(ik_steps):
            vel = mink.solve_ik(configuration, tasks, dt, "daqp")
            configuration.integrate_inplace(vel, dt)

        qpos_log.append(configuration.q.copy())

        # Compute errors
        mujoco.mj_forward(model, data)
        left_id = model.body("left_wrist_yaw_link").id
        right_id = model.body("right_wrist_yaw_link").id
        pos_errors_l.append(
            np.linalg.norm(data.xpos[left_id] - ref_data["left_pos"][frame])
        )
        pos_errors_r.append(
            np.linalg.norm(data.xpos[right_id] - ref_data["right_pos"][frame])
        )

    return {
        "qpos_log": np.array(qpos_log),
        "pos_errors_l": np.array(pos_errors_l),
        "pos_errors_r": np.array(pos_errors_r),
    }


def _pose_to_se3(pos: np.ndarray, quat_wxyz: np.ndarray):
    """Convert position + wxyz quaternion to an SE3 pose for mink."""
    import mink  # noqa: PLC0415

    q_xyzw = quat_wxyz[[1, 2, 3, 0]]
    rot = Rotation.from_quat(q_xyzw).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return mink.SE3.from_matrix(T)
