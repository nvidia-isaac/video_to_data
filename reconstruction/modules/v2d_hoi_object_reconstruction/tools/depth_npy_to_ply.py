#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Convert a depth .npy file to a .ply point cloud viewable in CloudCompare.

Requires camera intrinsics to back-project depth to 3D.

Usage:
  # With intrinsics from a config yaml
  python depth_npy_to_ply.py depth/left000000.npy --config /path/to/config.yaml --output /tmp/out.ply

  # With manual intrinsics (Hawk camera defaults)
  python depth_npy_to_ply.py depth/left000000.npy --fx 852.4 --fy 852.4 --cx 946.4 --cy 556.8

  # Pair with RGB image for colored point cloud
  python depth_npy_to_ply.py depth/left000000.npy --config config.yaml --rgb left/left000000.png
"""

import argparse
import numpy as np
import sys
from pathlib import Path


def depth_to_pointcloud(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                        rgb: np.ndarray | None = None, max_depth: float = 5.0) -> tuple:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth > 0) & (depth < max_depth)
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    pts = np.stack([x, y, z], axis=-1)
    colors = None
    if rgb is not None:
        colors = rgb[valid]
    return pts, colors


def save_ply(path: str, pts: np.ndarray, colors: np.ndarray | None = None) -> None:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(path, pcd)
    print(f"Saved {len(pts)} points to {path}")


def load_intrinsics_from_config(config_path: str) -> tuple[float, float, float, float]:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    K = cfg['camera_config']['intrinsic']
    return K[0], K[4], K[2], K[5]  # fx, fy, cx, cy


def main():
    parser = argparse.ArgumentParser(description="Convert depth .npy to .ply for CloudCompare")
    parser.add_argument('depth', type=Path, help='Depth .npy file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output .ply path (default: same name as input)')
    parser.add_argument('--config', type=str, default=None,
                        help='Config YAML to read intrinsics from')
    parser.add_argument('--fx', type=float, default=852.41)
    parser.add_argument('--fy', type=float, default=852.41)
    parser.add_argument('--cx', type=float, default=946.45)
    parser.add_argument('--cy', type=float, default=556.80)
    parser.add_argument('--rgb', type=str, default=None,
                        help='RGB image to colorize the point cloud')
    parser.add_argument('--max-depth', type=float, default=5.0,
                        help='Max depth in meters to include (default: 5.0)')
    args = parser.parse_args()

    if not args.depth.exists():
        print(f"ERROR: {args.depth} not found")
        sys.exit(1)

    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    if args.config:
        fx, fy, cx, cy = load_intrinsics_from_config(args.config)
        print(f"Intrinsics from config: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    depth = np.load(args.depth)
    print(f"Depth shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}] m")

    # Scale intrinsics to match depth resolution (depth may be downscaled from original)
    # Infer scale from depth width vs assumed full-res width (1920 for Hawk camera)
    assumed_full_width = 1920.0
    scale = depth.shape[1] / assumed_full_width
    fx *= scale; fy *= scale; cx *= scale; cy *= scale
    print(f"Depth scale factor: {scale:.3f} → scaled intrinsics: fx={fx:.2f} cx={cx:.2f}")

    rgb = None
    if args.rgb:
        import cv2
        rgb = cv2.cvtColor(cv2.imread(args.rgb), cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != depth.shape[:2]:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)

    pts, colors = depth_to_pointcloud(depth, fx, fy, cx, cy, rgb=rgb, max_depth=args.max_depth)

    output = args.output or str(args.depth.with_suffix('.ply'))
    save_ply(output, pts, colors)


if __name__ == '__main__':
    main()
