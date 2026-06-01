# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pose-math and image-conversion utilities for reconstruction scripts."""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def axis_angle_to_matrix(ax, ay, az, angle_degrees):
    """Convert axis-angle representation to a 3×3 rotation matrix."""
    axis = np.array([ax, ay, az])
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    rotvec = (axis / norm) * np.deg2rad(angle_degrees)
    return Rotation.from_rotvec(rotvec).as_matrix()


def cam_to_world_to_matrix(c2w):
    """Convert a CuSFM camera_to_world dict to a 4×4 numpy matrix."""
    aa = c2w['axis_angle']
    t  = c2w['translation']
    R  = axis_angle_to_matrix(aa['x'], aa['y'], aa['z'], aa['angle_degrees'])
    T  = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [t['x'], t['y'], t['z']]
    return T


def build_timestamp_to_seq_idx(frames_meta_path: Path, camera: str = 'left') -> dict:
    """Return {timestamp_us (int) → sequential job-folder index (0-based)}.

    Matches the ordering produced by prepare_FP_folder.py: stereo pairs sorted
    by synced_sample_id, keeping only pairs that have both cameras.
    """
    with open(frames_meta_path) as f:
        meta = json.load(f)
    cam_params = meta['camera_params_id_to_camera_params']
    left_sids  = {}
    right_sids = set()
    for kf in meta['keyframes_metadata']:
        cam_id = kf['camera_params_id']
        sid    = int(kf['synced_sample_id'])
        sensor = cam_params[cam_id]['sensor_meta_data']['sensor_name']
        if f'front_stereo_camera_{camera}' in sensor:
            left_sids[sid] = int(kf['timestamp_microseconds'])
        elif 'front_stereo_camera_right' in sensor:
            right_sids.add(sid)
    common_sids = sorted(set(left_sids) & right_sids)
    return {left_sids[sid]: i for i, sid in enumerate(common_sids)}


def save_as_png(src: Path, dst: Path):
    """Write src image to dst as PNG, symlinking if already PNG."""
    try:
        dst.unlink()
    except FileNotFoundError:
        pass
    if src.suffix.lower() == '.png':
        dst.symlink_to(os.path.relpath(src, dst.parent))
    else:
        Image.open(src).save(dst, format='PNG')
