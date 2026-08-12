#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone visualization of camera poses and point cloud from keyframes.yml.

No Docker / heavy ML deps required. Only needs:
  trimesh, pyglet<2, ruamel.yaml, numpy, open3d (optional)

Usage:
  python visualize_reconstruction_standalone.py /path/to/output/dir
  python visualize_reconstruction_standalone.py /path/to/output/dir --export scene.glb
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh.scene import Scene
from ruamel.yaml import YAML


def load_keyframes(keyframes_path: Path) -> dict:
    yaml = YAML()
    with open(keyframes_path, 'r') as f:
        return yaml.load(f)


def create_camera_geometry(pose: np.ndarray, scale: float = 0.03) -> list:
    geometries = []

    # Axis lines: red=X, green=Y, blue=Z
    for i, color in enumerate([[255, 0, 0], [0, 255, 0], [0, 0, 255]]):
        direction = pose[:3, i] * scale
        pts = np.array([pose[:3, 3], pose[:3, 3] + direction])
        line = trimesh.load_path(pts)
        line.colors = np.tile(color + [255], (len(line.entities), 1))
        geometries.append(line)

    # Camera frustum
    origin = pose[:3, 3]
    z = pose[:3, 2] * scale * 1.5
    center = origin + z
    x_axis, y_axis = pose[:3, 0], pose[:3, 1]
    hw = scale * 0.4
    corners = [
        center + x_axis * hw + y_axis * hw,
        center - x_axis * hw + y_axis * hw,
        center - x_axis * hw - y_axis * hw,
        center + x_axis * hw - y_axis * hw,
    ]
    for c in corners:
        line = trimesh.load_path(np.array([origin, c]))
        line.colors = np.array([[255, 255, 0, 255]])
        geometries.append(line)
    for i in range(4):
        line = trimesh.load_path(np.array([corners[i], corners[(i + 1) % 4]]))
        line.colors = np.array([[255, 255, 0, 255]])
        geometries.append(line)

    return geometries


def load_pointcloud(output_dir: Path, eps: float) -> trimesh.PointCloud | None:
    """Load pcd_normalized.ply — the object point cloud from the reconstruction pipeline."""
    try:
        import open3d as o3d
    except ImportError:
        print("open3d not installed — skipping point cloud")
        return None

    # Use only the normalized object point cloud; skip octree/voxel grid files
    candidate = output_dir / "pcd_normalized.ply"
    if not candidate.exists():
        print(f"pcd_normalized.ply not found in {output_dir} — skipping point cloud")
        return None

    print(f"Loading point cloud: {candidate}")
    merged = o3d.io.read_point_cloud(str(candidate))

    if eps > 0:
        merged = merged.voxel_down_sample(eps)

    pts = np.asarray(merged.points)
    colors = (np.asarray(merged.colors) * 255).astype(np.uint8) if merged.has_colors() else None
    return trimesh.PointCloud(vertices=pts, colors=colors)


def build_scene(keyframes: dict, scale: float, eps: float, output_dir: Path) -> Scene:
    scene = Scene()

    keys = list(keyframes.keys())
    print(f"Selected {len(keys)} keyframes")

    for k in keys:
        pose = np.array(keyframes[k]['cam_in_ob']).reshape(4, 4)
        for g in create_camera_geometry(pose, scale=scale):
            scene.add_geometry(g)

    pcd = load_pointcloud(output_dir, eps)
    if pcd is not None and len(pcd.vertices) > 0:
        print(f"Loaded point cloud: {len(pcd.vertices)} points")
        scene.add_geometry(pcd)
    else:
        # Placeholder box when no point cloud available
        scene.add_geometry(trimesh.primitives.Box(extents=[0.1] * 3))

    return scene


def main():
    parser = argparse.ArgumentParser(description="Standalone reconstruction visualizer")
    parser.add_argument('output_dir', type=Path, help='Directory containing keyframes.yml')
    parser.add_argument('--scale', type=float, default=0.03, help='Camera frustum scale')
    parser.add_argument('--eps', type=float, default=0.01, help='Point cloud voxel downsample size')
    parser.add_argument('--export', type=str, default=None,
                        help='Export to file instead of opening viewer (e.g. scene.glb, scene.html)')
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    keyframes_path = output_dir / 'keyframes.yml'
    if not keyframes_path.exists():
        print(f"ERROR: keyframes.yml not found in {output_dir}")
        sys.exit(1)

    keyframes = load_keyframes(keyframes_path)
    scene = build_scene(keyframes, args.scale, args.eps, output_dir)

    if args.export:
        scene.export(args.export)
        print(f"Exported to {args.export}")
    else:
        print("Opening viewer  (left-drag: rotate, right-drag: pan, scroll: zoom)")
        scene.show()


if __name__ == '__main__':
    main()
