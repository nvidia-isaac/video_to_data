# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Convert a folder of Transform3d (camera-to-world) JSONs into a PLY of camera
frustum wireframes. Open this PLY alongside the point cloud in MeshLab (or
any PLY viewer) to see where each camera is pointing.

Usage:
    python modules/v2d_droid_slam/tools/poses_to_frustums.py \
        --poses_folder    data/clean/agent_90_undist/output/droid_slam/poses \
        --intrinsics_path data/clean/agent_90_undist/undistorted_intrinsics.json \
        --output_path     data/clean/agent_90_undist/output/droid_slam/cameras.ply \
        [--frustum_size 0.05]   # near-plane distance in scene units
        [--stride 1]            # skip frames

Open in MeshLab (drag both files into the same window):
    meshlab data/clean/agent_90_undist/output/droid_slam/pointcloud.ply \
            data/clean/agent_90_undist/output/droid_slam/cameras.ply
"""
import argparse
import json
import os
import sys

import numpy as np

# Make `v2d.common.datatypes` importable without installing anything.
_REPO_MODULES = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_MODULES, "v2d_common"))
from datatypes import CameraIntrinsics, Transform3d  # noqa: E402


def _frustum_vertices(K: np.ndarray, T_cw: np.ndarray, near: float,
                      width: int, height: int) -> np.ndarray:
    """Return 5 vertices (world frame): camera centre + 4 near-plane corners."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    corners_pix = np.array([
        [0, 0], [width, 0], [width, height], [0, height],
    ], dtype=np.float64)
    rays_cam = np.stack([
        (corners_pix[:, 0] - cx) / fx,
        (corners_pix[:, 1] - cy) / fy,
        np.ones(4),
    ], axis=1) * near
    pts_cam = np.concatenate([np.zeros((1, 3)), rays_cam], axis=0)  # (5, 3)
    pts_world = (T_cw[:3, :3] @ pts_cam.T + T_cw[:3, 3:4]).T
    return pts_world


_FRUSTUM_EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4),  # apex → corners
    (1, 2), (2, 3), (3, 4), (4, 1),  # near-plane rectangle
]


def _rgb_for_index(i: int, n: int) -> tuple[int, int, int]:
    """Linearly interpolate red → green → blue across the trajectory."""
    t = i / max(n - 1, 1)
    if t < 0.5:
        r = int(255 * (1 - 2 * t))
        g = int(255 * (2 * t))
        b = 0
    else:
        r = 0
        g = int(255 * (2 * (1 - t)))
        b = int(255 * (2 * t - 1))
    return r, g, b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses_folder", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--frustum_size", type=float, default=0.05,
                        help="Near-plane distance in scene units")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    intr = CameraIntrinsics.load(args.intrinsics_path)
    K = intr.to_matrix()
    pose_files = sorted(f for f in os.listdir(args.poses_folder) if f.endswith(".json"))
    pose_files = pose_files[::args.stride]
    if not pose_files:
        raise FileNotFoundError(f"No pose JSONs in {args.poses_folder}")

    verts, edges, edge_colors = [], [], []
    n = len(pose_files)
    for i, fn in enumerate(pose_files):
        T_cw = Transform3d.load(os.path.join(args.poses_folder, fn)).to_matrix()
        v = _frustum_vertices(K, T_cw, args.frustum_size, intr.width, intr.height)
        base = i * 5
        verts.append(v)
        rgb = _rgb_for_index(i, n)
        for a, b in _FRUSTUM_EDGES:
            edges.append((base + a, base + b))
            edge_colors.append(rgb)

    verts_arr = np.concatenate(verts, axis=0).astype(np.float32)
    n_v = verts_arr.shape[0]
    n_e = len(edges)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        f.write(
            f"ply\nformat ascii 1.0\n"
            f"element vertex {n_v}\n"
            "property float x\nproperty float y\nproperty float z\n"
            f"element edge {n_e}\n"
            "property int vertex1\nproperty int vertex2\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for x, y, z in verts_arr:
            f.write(f"{x} {y} {z}\n")
        for (a, b), (r, g, bl) in zip(edges, edge_colors):
            f.write(f"{a} {b} {r} {g} {bl}\n")

    print(f"wrote {args.output_path}: {n} frustums, {n_v} vertices, {n_e} edges")
    print(f"View with: meshlab <pointcloud.ply> {args.output_path}")


if __name__ == "__main__":
    main()
