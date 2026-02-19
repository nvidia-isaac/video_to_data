# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Utility functions for the V2P environment."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

from robotic_grounding.retarget.data_logger import ManoSharpaData


def interpolate_robot_motion_data(
    motion_data: ManoSharpaData,
    target_fps: float,
) -> ManoSharpaData:
    """Interpolate the robot motion data to the target FPS.

    Now only supports interpolation of the following fields:
    - object_body_position -> linear interpolation
    - object_body_wxyz -> Slerp
    - object_articulation -> linear interpolation
    - robot_right_wrist_position -> linear interpolation
    - robot_right_wrist_wxyz -> Slerp interpolation
    - robot_right_finger_joints -> linear interpolation
    - robot_left_wrist_position -> linear interpolation
    - robot_left_wrist_wxyz -> Slerp interpolation
    - robot_left_finger_joints -> linear interpolation
    - robot_right_frames -> linear interpolation for position, Slerp for orientation
    - robot_left_frames -> linear interpolation for position, Slerp for orientation
    TODO: we may need to improve this function for more general cases.

    Args:
        motion_data: The motion data to interpolate.
        target_fps: The target FPS to interpolate to.

    Returns:
        The interpolated motion data.
    """

    # --- Helper Functions ---
    def get_timestamps() -> tuple[np.ndarray, np.ndarray]:
        """Get the timestamps for the motion data."""
        n_frames = len(motion_data.robot_right_wrist_position)
        src = np.arange(n_frames) / motion_data.fps
        tgt = np.linspace(0, src[-1], int(src[-1] * target_fps))
        return src, tgt

    def interp_linear(data: list[float]) -> list[float]:
        """Vectorized linear interpolation for any shape (T, ...)."""
        data_arr = np.asarray(data)
        return interp1d(src_times, data_arr, kind="linear", axis=0)(tgt_times).tolist()

    def interp_slerp(quat_data: list[list[float]]) -> list[list[float]]:
        """Slerp for a single time sequence of quaternions (T, 4)."""
        quats = np.asarray(quat_data)
        rotations = Rotation.from_quat(quats, scalar_first=True)
        slerp = Slerp(src_times, rotations)
        return slerp(tgt_times).as_quat(scalar_first=True).tolist()

    def interp_slerp_batch(
        quat_data: list[list[list[float]]],
    ) -> list[list[list[float]]]:
        """Slerp for batched quaternions (T, N, 4). Loops over N."""
        data_arr = np.asarray(quat_data)
        # Shape is (Time, N, 4) -> Iterate over N
        n_items = data_arr.shape[1]
        results = []
        for i in range(n_items):
            # Extract (Time, 4) for the i-th item
            results.append(interp_slerp(data_arr[:, i, :]))
        # Stack back to (Time, N, 4) and convert to list
        return np.array(results).transpose(1, 0, 2).tolist()

    def interp_frames(frame_data: list[list[list[float]]]) -> list[list[list[float]]]:
        """Handles (T, N, 7) arrays: Split Pos (linear) and Rot (Slerp)."""
        arr = np.asarray(frame_data)
        # 1. Linear interp on position (first 3 cols) - Vectorized
        pos_interp = np.array(interp_linear(arr[:, :, :3]))
        # 2. Slerp on rotation (last 4 cols) - Batched
        rot_interp = np.array(interp_slerp_batch(arr[:, :, 3:]))
        # 3. Combine
        return np.concatenate([pos_interp, rot_interp], axis=2).tolist()

    # --- Execution ---

    # 0. Setup Times
    src_times, tgt_times = get_timestamps()

    # 1. Simple Linear Fields
    motion_data.object_articulation = interp_linear(motion_data.object_articulation)
    motion_data.robot_right_finger_joints = interp_linear(
        motion_data.robot_right_finger_joints
    )
    motion_data.robot_left_finger_joints = interp_linear(
        motion_data.robot_left_finger_joints
    )
    motion_data.robot_right_wrist_position = interp_linear(
        motion_data.robot_right_wrist_position
    )
    motion_data.robot_left_wrist_position = interp_linear(
        motion_data.robot_left_wrist_position
    )

    # 2. Simple Rotation Fields (Slerp)
    motion_data.robot_right_wrist_wxyz = interp_slerp(
        motion_data.robot_right_wrist_wxyz
    )
    motion_data.robot_left_wrist_wxyz = interp_slerp(motion_data.robot_left_wrist_wxyz)

    # 3. Object Bodies (Batched Position & Rotation)
    # Note: interp_linear handles (Time, N, 3) automatically without loops
    motion_data.object_body_position = interp_linear(motion_data.object_body_position)
    motion_data.object_body_wxyz = interp_slerp_batch(motion_data.object_body_wxyz)

    # 4. Frames (Batched Mixed Data)
    motion_data.robot_right_frames = interp_frames(motion_data.robot_right_frames)
    motion_data.robot_left_frames = interp_frames(motion_data.robot_left_frames)

    return motion_data
