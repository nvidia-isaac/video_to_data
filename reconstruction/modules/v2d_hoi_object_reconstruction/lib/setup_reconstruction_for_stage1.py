#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Set up Stage-1 reconstruction directory from CuSFM keyframes.

Selects SfM keyframes with seq_idx <= stage1_end_frame and writes:
  left/, right/, depth/ (symlinks), poses/, keyframes.yml

Usage:
    python setup_reconstruction_for_stage1.py \\
        --job_dir          /data/.../              \\
        --frames_meta      /data/.../frames_meta.json \\
        --sfm_keyframes    /data/.../sfm/keyframes/frames_meta.json \\
        --stage1_end_frame 969 \\
        --depth_dir        /data/.../depth \\
        --output_dir       /data/.../stage1_recon
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

from v2d_hoi_object_reconstruction.lib.recon_utils import (
    axis_angle_to_matrix,
    cam_to_world_to_matrix,
    build_timestamp_to_seq_idx,
    save_as_png,
)


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_sfm_keyframes_stage1(sfm_path: Path, ts_us_to_seq_idx: dict,
                               stage1_end_frame: int, camera: str) -> list:
    """Return [(seq_idx, T_world_from_cam)] for Stage-1 SfM keyframes."""
    with open(sfm_path) as f:
        data = json.load(f)

    cam_substr = f'front_stereo_camera_{camera}'
    result = []
    unmatched = 0

    for kf in data['keyframes_metadata']:
        if cam_substr not in kf.get('image_name', ''):
            continue
        ts_us   = int(kf['timestamp_microseconds'])
        seq_idx = ts_us_to_seq_idx.get(ts_us)
        if seq_idx is None:
            unmatched += 1
            continue
        if seq_idx <= stage1_end_frame:
            T = cam_to_world_to_matrix(kf['camera_to_world'])
            result.append((seq_idx, T))

    if unmatched:
        print(f"  WARNING: {unmatched} SfM keyframes had no timestamp match")
    return sorted(result, key=lambda x: x[0])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Set up Stage-1 reconstruction directory from SfM keyframes')
    parser.add_argument('--job_dir',          default=None,
                        help='Prepared job directory (used to derive --left_dir/--right_dir '
                             'if those are not set explicitly)')
    parser.add_argument('--left_dir',         default=None,
                        help='Left image directory ({seq_idx:06d}.jpg). '
                             'Defaults to <job_dir>/left if --job_dir is given.')
    parser.add_argument('--right_dir',        default=None,
                        help='Right image directory ({seq_idx:06d}.jpg). '
                             'Defaults to <job_dir>/right if --job_dir is given.')
    parser.add_argument('--frames_meta',      required=True,
                        help='frames_meta.json from mapping_data (all frames, no poses)')
    parser.add_argument('--sfm_keyframes',    required=True,
                        help='CuSFM frames_meta.json with camera-to-world poses')
    stage1_group = parser.add_mutually_exclusive_group(required=True)
    stage1_group.add_argument('--stage1_end_frame', type=int,
                        help='Last sequential job-folder index of Stage 1 (0-based, inclusive)')
    stage1_group.add_argument('--stage1_end_timestamp', type=int,
                        help='Nanosecond timestamp of the last Stage-1 frame '
                             '(matches image filename in front_stereo_camera_left/)')
    parser.add_argument('--depth_dir',        required=True,
                        help='Depth directory (depth/{seq_idx:06d}.png files)')
    parser.add_argument('--masks_dir',        default=None,
                        help='Masks directory ({seq_idx:06d}.png files). '
                             'If provided, creates masks/ symlinks in output_dir.')
    parser.add_argument('--output_dir',       required=True,
                        help='Output Stage-1 reconstruction directory')
    parser.add_argument('--camera',           default='left', choices=['left', 'right'])
    args = parser.parse_args()

    if args.left_dir is None and args.right_dir is None and args.job_dir is None:
        parser.error('Provide --job_dir or both --left_dir and --right_dir')
    left_dir  = Path(args.left_dir)  if args.left_dir  else Path(args.job_dir) / 'left'
    right_dir = Path(args.right_dir) if args.right_dir else Path(args.job_dir) / 'right'

    job_dir    = left_dir.parent  # kept for any legacy references
    output_dir = Path(args.output_dir)
    depth_dir  = Path(args.depth_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ('left', 'right', 'poses', 'masks', 'depth'):
        (output_dir / sub).mkdir(exist_ok=True)

    print("Building timestamp → seq_idx index...")
    ts_us_to_seq_idx = build_timestamp_to_seq_idx(Path(args.frames_meta), args.camera)

    if args.stage1_end_frame is not None:
        stage1_end_frame = args.stage1_end_frame
    else:
        ts_us = args.stage1_end_timestamp // 1000
        stage1_end_frame = ts_us_to_seq_idx.get(ts_us)
        if stage1_end_frame is None:
            raise ValueError(
                f"--stage1_end_timestamp {args.stage1_end_timestamp} "
                f"(={ts_us} µs) not found in frames_meta.json")
        print(f"  stage1_end_timestamp {args.stage1_end_timestamp} → seq_idx {stage1_end_frame}")

    print(f"Loading SfM keyframes (stage1_end_frame={stage1_end_frame})...")
    keyframes = load_sfm_keyframes_stage1(
        Path(args.sfm_keyframes), ts_us_to_seq_idx, stage1_end_frame, args.camera)
    print(f"  Stage-1 keyframes selected: {len(keyframes)}")

    frames = []
    skipped = 0
    for out_idx, (seq_idx, T_world_from_cam) in enumerate(keyframes):
        img_src = left_dir / f'{seq_idx:06d}.jpg'
        if not img_src.exists():
            skipped += 1
            continue
        img_right = right_dir / f'{seq_idx:06d}.jpg'
        if not img_right.exists():
            img_right = None
        # cam_in_ob = T_world_from_cam (object is at CuSFM world origin for stage 1)
        frames.append((out_idx, seq_idx, T_world_from_cam, img_src, img_right))

    # Write poses and keyframes.yml
    kf_yml = {}
    for out_idx, seq_idx, cam_in_ob, _, _ in frames:
        stem = f'left{out_idx:06d}'
        with open(output_dir / 'poses' / f'{stem}.json', 'w') as pf:
            json.dump(cam_in_ob.tolist(), pf)
        kf_yml[f'keyframe_{stem}'] = {'cam_in_ob': cam_in_ob.flatten().tolist()}

    # Convert images to PNG in parallel
    def convert(item):
        out_idx, seq_idx, _, img_src, img_right = item
        save_as_png(img_src, output_dir / 'left' / f'left{out_idx:06d}.png')
        if img_right:
            save_as_png(img_right, output_dir / 'right' / f'right{out_idx:06d}.png')

    n_workers = min(os.cpu_count(), 16)
    print(f"Converting {len(frames)} frames to PNG (workers={n_workers})...")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for fut in as_completed({pool.submit(convert, f): f for f in frames}):
            fut.result()

    with open(output_dir / 'keyframes.yml', 'w') as f:
        yaml.dump(kf_yml, f, default_flow_style=None, sort_keys=False)

    # Depth symlinks: depth_dir/{seq_idx:06d}.png → output_dir/depth/left{out_idx:06d}.png
    # Use relative symlinks so they resolve correctly both inside and outside Docker.
    depth_link_dir = output_dir / 'depth'
    for out_idx, seq_idx, _, _, _ in frames:
        src = depth_dir / f'{seq_idx:06d}.png'
        dst = depth_link_dir / f'left{out_idx:06d}.png'
        try:
            dst.unlink()
        except FileNotFoundError:
            pass
        dst.symlink_to(os.path.relpath(src, dst.parent))

    # Mask symlinks: masks_dir/0/{seq_idx:06d}.png → output_dir/masks/left{out_idx:06d}.png
    if args.masks_dir:
        masks_src_dir  = Path(args.masks_dir) / '0'
        masks_link_dir = output_dir / 'masks'
        for out_idx, seq_idx, _, _, _ in frames:
            src = masks_src_dir / f'{seq_idx:06d}.png'
            dst = masks_link_dir / f'left{out_idx:06d}.png'
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            dst.symlink_to(os.path.relpath(src, dst.parent))
        print(f"  Mask symlinks: {len(frames)}")

    print(f"\nDone.")
    print(f"  Frames written: {len(frames)}  (skipped: {skipped})")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
