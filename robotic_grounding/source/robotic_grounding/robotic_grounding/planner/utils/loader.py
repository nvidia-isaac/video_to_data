"""V2P retargeted-motion loading helpers shared across planner backends."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def interpolate_robot_motion_data(motion_data: Any, target_fps: float) -> Any:
    """Resample a V2P retargeted motion's time-series fields to ``target_fps``.

    Linear interpolation for positions / joint angles / object articulation,
    SLERP for wrist + per-body object quaternions, and contact-aware linear
    interpolation that zeroes intervals where either neighbour is inactive.
    """
    n_frames = len(motion_data.robot_right_wrist_position)
    src_times = np.arange(n_frames) / motion_data.fps
    tgt_times = np.linspace(0, src_times[-1], int(src_times[-1] * target_fps))

    def _linear(data: Any) -> Any:
        return interp1d(src_times, np.asarray(data), kind="linear", axis=0)(
            tgt_times
        ).tolist()

    def _slerp(quat_data: Any) -> Any:
        quats = np.asarray(quat_data)
        return (
            Slerp(src_times, Rotation.from_quat(quats, scalar_first=True))(tgt_times)
            .as_quat(scalar_first=True)
            .tolist()
        )

    def _slerp_batch(quat_data: Any) -> Any:
        arr = np.asarray(quat_data)
        results = [_slerp(arr[:, i, :]) for i in range(arr.shape[1])]
        return np.array(results).transpose(1, 0, 2).tolist()

    def _frames(frame_data: Any) -> Any:
        arr = np.asarray(frame_data)
        pos = np.array(_linear(arr[:, :, :3]))
        rot = np.array(_slerp_batch(arr[:, :, 3:]))
        return np.concatenate([pos, rot], axis=2).tolist()

    def _contact_linear(data: Any, part_ids: Any) -> Any:
        if not data:
            return []
        H, N = len(data), len(data[0])
        arr = np.concatenate(
            [np.asarray(data), np.asarray(part_ids).reshape(H, N, 1)], axis=-1
        )
        interp_result = interp1d(src_times, arr, kind="linear", axis=0)(tgt_times)
        nonzero_src = np.abs(arr) > 1e-8
        idx_lo = np.clip(
            np.searchsorted(src_times, tgt_times, side="right") - 1,
            0,
            len(src_times) - 1,
        )
        idx_hi = np.minimum(idx_lo + 1, len(src_times) - 1)
        mask = nonzero_src[idx_lo] & nonzero_src[idx_hi]
        return np.where(mask, interp_result, 0.0).tolist()

    motion_data.object_articulation = _linear(motion_data.object_articulation)
    motion_data.robot_right_finger_joints = _linear(
        motion_data.robot_right_finger_joints
    )
    motion_data.robot_left_finger_joints = _linear(motion_data.robot_left_finger_joints)
    motion_data.robot_right_wrist_position = _linear(
        motion_data.robot_right_wrist_position
    )
    motion_data.robot_left_wrist_position = _linear(
        motion_data.robot_left_wrist_position
    )
    motion_data.robot_right_wrist_wxyz = _slerp(motion_data.robot_right_wrist_wxyz)
    motion_data.robot_left_wrist_wxyz = _slerp(motion_data.robot_left_wrist_wxyz)
    motion_data.object_body_position = _linear(motion_data.object_body_position)
    motion_data.object_body_wxyz = _slerp_batch(motion_data.object_body_wxyz)
    motion_data.robot_right_frames = _frames(motion_data.robot_right_frames)
    motion_data.robot_left_frames = _frames(motion_data.robot_left_frames)

    for side in ("right", "left"):
        part_ids = getattr(motion_data, f"mano_{side}_object_contact_part_ids", [])
        for field in (
            "link_contact_positions",
            "object_contact_positions",
            "object_contact_normals",
        ):
            attr = f"mano_{side}_{field}"
            val = getattr(motion_data, attr, [])
            if val:
                setattr(motion_data, attr, _contact_linear(val, part_ids))

    return motion_data
