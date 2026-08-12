#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Fuse depth maps into a single world-frame point cloud using camera poses.

Two modes:
  keyframes (default) — uses keyframes.yml, object-only depth from depth/
  all-frames          — uses poses/*.json, full-scene depth from depth/

Usage:
  # Keyframes only (object point cloud)
  python fuse_depth_to_pointcloud.py data/output/2026-02-18_12-21-20_bowl/

  # All frames, full scene
  python fuse_depth_to_pointcloud.py data/output/2026-02-18_12-21-20_bowl/ --all-frames

  # All frames, object only
  python fuse_depth_to_pointcloud.py data/output/2026-02-18_12-21-20_bowl/ --all-frames --mask

  # Custom options
  python fuse_depth_to_pointcloud.py data/output/2026-02-18_12-21-20_bowl/ \\
      --config hoi_object_reconstruction/data/configs/theseus_optimizer_hawk.yaml \\
      --output /tmp/fused.ply --max-depth 3.0 --voxel-size 0.005
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_keyframe_poses(keyframes_path: Path) -> dict[str, np.ndarray]:
    """Return {frame_id: 4x4 c2w} from keyframes.yml."""
    from ruamel.yaml import YAML
    yaml = YAML()
    with open(keyframes_path) as f:
        kf = yaml.load(f)
    return {
        k.replace('keyframe_', ''): np.array(v['cam_in_ob']).reshape(4, 4)
        for k, v in kf.items()
    }


def load_all_poses(poses_dir: Path) -> dict[str, np.ndarray]:
    """Return {frame_id: 4x4 c2w} from poses/*.json."""
    poses = {}
    for p in sorted(poses_dir.glob('*.json')):
        frame_id = p.stem  # e.g. left000000
        poses[frame_id] = np.array(json.loads(p.read_text()))
    return poses


def load_intrinsics_from_config(config_path: Path) -> tuple[float, float, float, float]:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    K = cfg['camera_config']['intrinsic']
    return K[0], K[4], K[2], K[5]  # fx, fy, cx, cy


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def depth_to_points(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                    mask: np.ndarray | None = None,
                    max_depth: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    valid = (depth > 0) & (depth < max_depth)
    if mask is not None:
        valid &= (mask > 127)
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1)  # (N, 4)
    return pts_cam, valid


def fuse_frames(poses: dict[str, np.ndarray], depth_dir: Path, rgb_dir: Path,
                mask_dir: Path | None, fx: float, fy: float, cx: float, cy: float,
                max_depth: float, use_mask: bool) -> o3d.geometry.PointCloud:
    merged = o3d.geometry.PointCloud()
    skipped = 0

    for frame_id, c2w in tqdm(poses.items(), desc="Fusing frames"):
        depth_path = depth_dir / f"{frame_id}.npy"
        rgb_path = rgb_dir / f"{frame_id}.png"

        if not depth_path.exists():
            skipped += 1
            continue

        depth = np.load(depth_path)
        h, w = depth.shape

        # Scale intrinsics to match depth resolution
        scale = w / 1920.0
        sfx, sfy = fx * scale, fy * scale
        scx, scy = cx * scale, cy * scale

        # Optional mask
        mask = None
        if use_mask and mask_dir and mask_dir.exists():
            mask_path = mask_dir / f"{frame_id}.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask.shape != depth.shape:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        pts_cam, valid = depth_to_points(depth, sfx, sfy, scx, scy,
                                         mask=mask, max_depth=max_depth)
        if len(pts_cam) == 0:
            skipped += 1
            continue

        pts_world = (c2w @ pts_cam.T).T[:, :3]

        colors = None
        if rgb_path.exists():
            rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != depth.shape:
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            colors = rgb[valid].astype(np.float64) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        merged += pcd

    if skipped:
        print(f"Skipped {skipped} frames (missing depth)")
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fuse depth maps into world-frame point cloud")
    parser.add_argument('output_dir', type=Path,
                        help='Reconstruction output directory (contains depth/, left/, poses/ or keyframes.yml)')
    parser.add_argument('--all-frames', action='store_true',
                        help='Use all frames via poses/*.json instead of keyframes.yml only')
    parser.add_argument('--config', type=Path, default=None,
                        help='Config YAML for camera intrinsics')
    parser.add_argument('--fx', type=float, default=852.41)
    parser.add_argument('--fy', type=float, default=852.41)
    parser.add_argument('--cx', type=float, default=946.45)
    parser.add_argument('--cy', type=float, default=556.80)
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output .ply path (default: output_dir/fused_depth.ply)')
    parser.add_argument('--max-depth', type=float, default=3.0,
                        help='Max depth in metres (default: 3.0)')
    parser.add_argument('--voxel-size', type=float, default=0.005,
                        help='Voxel downsample size in metres (default: 0.005, 0 to disable)')
    parser.add_argument('--mask', action='store_true',
                        help='Apply object mask (masks/) to keep only object points')
    parser.add_argument('--frame-start', type=int, default=None,
                        help='First frame index to include (0-based, inclusive)')
    parser.add_argument('--frame-end', type=int, default=None,
                        help='Last frame index to include (0-based, inclusive)')
    parser.add_argument('--skip-frames', type=int, nargs='+', default=None,
                        metavar='IDX', help='Frame indices to skip (e.g. --skip-frames 3 17)')
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    depth_dir = output_dir / 'depth'
    rgb_dir = output_dir / 'left'
    mask_dir = output_dir / 'masks'
    poses_dir = output_dir / 'poses'

    if not depth_dir.exists():
        print(f"ERROR: depth/ not found in {output_dir}")
        sys.exit(1)

    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    if args.config:
        fx, fy, cx, cy = load_intrinsics_from_config(args.config)
        print(f"Intrinsics from config: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    if args.all_frames:
        if not poses_dir.exists():
            print(f"ERROR: poses/ directory not found in {output_dir}")
            sys.exit(1)
        poses = load_all_poses(poses_dir)
        print(f"All-frames mode: {len(poses)} pose files in poses/")
    else:
        keyframes_path = output_dir / 'keyframes.yml'
        if not keyframes_path.exists():
            print(f"ERROR: keyframes.yml not found — use --all-frames to use poses/ instead")
            sys.exit(1)
        poses = load_keyframe_poses(keyframes_path)
        print(f"Keyframes mode: {len(poses)} keyframes")

    # Apply frame range and skip filters
    keys = list(poses.keys())
    start = args.frame_start or 0
    end = (args.frame_end + 1) if args.frame_end is not None else len(keys)
    keys = keys[start:end]

    skip_set = set(args.skip_frames) if args.skip_frames else set()
    if skip_set:
        skipped_keys = [keys[i] for i in skip_set if i < len(keys)]
        print(f"Skipping frames at indices {sorted(skip_set)}: {skipped_keys}")
        keys = [k for i, k in enumerate(keys) if i not in skip_set]

    if args.frame_start is not None or args.frame_end is not None or skip_set:
        poses = {k: poses[k] for k in keys}
        print(f"Frames selected: {len(poses)}")

    merged = fuse_frames(poses, depth_dir, rgb_dir, mask_dir,
                         fx, fy, cx, cy, args.max_depth, args.mask)

    print(f"Total points before downsample: {len(merged.points)}")

    if args.voxel_size > 0:
        merged = merged.voxel_down_sample(args.voxel_size)
        print(f"After voxel downsample ({args.voxel_size}m): {len(merged.points)} points")

    suffix = '_all_frames' if args.all_frames else '_keyframes'
    suffix += '_masked' if args.mask else '_full_scene'
    default_out = str(output_dir / f'fused_depth{suffix}.ply')
    out_path = args.output or default_out
    o3d.io.write_point_cloud(out_path, merged)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
