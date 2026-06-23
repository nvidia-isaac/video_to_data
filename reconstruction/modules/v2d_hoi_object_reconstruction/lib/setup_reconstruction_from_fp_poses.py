#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Set up 3D reconstruction directory from CuSFM keyframes filtered to scanning stages.

Selects CuSFM keyframes from both scanning stages (excluding transition frames)
and writes a single merged reconstruction directory with:
  - left/  right/  : PNG images
  - poses/         : per-frame camera-to-world JSON (4×4)
  - keyframes.yml  : cam_in_ob = T_cam_to_obj (camera-to-object)

For reconstruction we need T_camera_to_object. Since the object is static within
each stage, T_camera_to_world (CuSFM) serves as T_camera_to_object within that stage.
However, the object is physically rotated between stage 1 and stage 2, so stage 2
camera poses must be brought into the stage 1 object frame:

  Stage 1: cam_in_ob = T_cam_to_world          (CuSFM directly)
  Stage 2: cam_in_ob = inv(T_pose2_from_pose1) @ T_cam_to_world

T_pose2_from_pose1 is loaded from poses_world.json (stage_analysis field).

Frame selection uses CuSFM keyframes matched to frame_id by timestamp, filtered
to stage 1 and stage 2 bounds from poses_world.json.

Usage:
    python setup_reconstruction_from_fp_poses.py \\
        --poses_world      /path/to/poses_world.json \\
        --frame_metadata   /path/to/mapping_data/frame_metadata.jsonl \\
        --mapping_data_dir /path/to/mapping_data \\
        --sfm_keyframes    /path/to/kpmap/keyframes/frames_meta.json \\
        --output_dir       /path/to/output
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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_stage_fid_sets(poses_world_path: Path):
    """Return (stage1_fids, stage2_fids, T_obj_stage1, T_obj_stage2).

    T_obj_stage1 and T_obj_stage2 are representative FP-derived object-to-world
    poses for each stage. cam_in_ob for each camera is:
        cam_in_ob = inv(T_world_from_obj_stage) @ T_cam_world
    """
    with open(poses_world_path) as f:
        d = json.load(f)
    sa = d['stage_analysis']

    def fid_set(key):
        start = sa[key]['start_frame']
        end   = sa[key]['end_frame']
        return set(fr['frame_id'] for fr in d['frames'] if start <= fr['frame_id'] <= end)

    T_obj_stage1 = np.array(sa['stage1']['T_world_from_obj'])
    T_obj_stage2 = np.array(sa['stage2']['T_world_from_obj'])

    return fid_set('stage1'), fid_set('stage2'), T_obj_stage1, T_obj_stage2


def load_sfm_keyframes(sfm_keyframes_path: Path, ts_us_to_seq_idx: dict,
                       valid_seq_set: set, camera: str) -> list:
    """
    Return list of (seq_idx, T_world_from_cam) for SfM keyframes within
    valid_seq_set, matched by timestamp_us. Sorted by seq_idx.
    """
    with open(sfm_keyframes_path) as f:
        frames_meta = json.load(f)

    cam_substr = f'front_stereo_camera_{camera}'
    result = []
    unmatched = 0

    for kf in frames_meta['keyframes_metadata']:
        if cam_substr not in kf.get('image_name', ''):
            continue
        ts_us   = int(kf['timestamp_microseconds'])
        seq_idx = ts_us_to_seq_idx.get(ts_us)
        if seq_idx is None:
            unmatched += 1
            continue
        if seq_idx in valid_seq_set:
            T = cam_to_world_to_matrix(kf['camera_to_world'])
            result.append((seq_idx, T))

    if unmatched:
        print(f"  WARNING: {unmatched} SfM keyframes had no timestamp match")

    return sorted(result, key=lambda x: x[0])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Set up merged reconstruction directory using CuSFM poses filtered to stages')
    parser.add_argument('--poses_world',   required=True,
                        help='poses_world.json: provides stage bounds and T_pose2_from_pose1')
    parser.add_argument('--frames_meta',   required=True,
                        help='frames_meta.json from mapping_data (covers all frames)')
    parser.add_argument('--job_dir',       default=None,
                        help='Prepared job directory (used to derive --left_dir/--right_dir '
                             'if those are not set explicitly)')
    parser.add_argument('--left_dir',      default=None,
                        help='Left image directory ({seq_idx:06d}.jpg). '
                             'Defaults to <job_dir>/left if --job_dir is given.')
    parser.add_argument('--right_dir',     default=None,
                        help='Right image directory ({seq_idx:06d}.jpg). '
                             'Defaults to <job_dir>/right if --job_dir is given.')
    parser.add_argument('--sfm_keyframes', required=True,
                        help='frames_meta.json with SfM camera-to-world poses')
    parser.add_argument('--output_dir',    required=True,
                        help='Output reconstruction directory')
    parser.add_argument('--depth_dir',     default=None,
                        help='Depth directory for symlinks (depth/{seq_idx:06d}.png). '
                             'If provided, creates depth/ symlinks in output_dir.')
    parser.add_argument('--masks_dir',     default=None,
                        help='Masks directory for symlinks ({seq_idx:06d}.png). '
                             'If provided, creates masks/ symlinks in output_dir.')
    parser.add_argument('--camera',        default='left', choices=['left', 'right'],
                        help='Primary camera side (default: left)')
    args = parser.parse_args()

    if args.left_dir is None and args.right_dir is None and args.job_dir is None:
        parser.error('Provide --job_dir or both --left_dir and --right_dir')
    left_dir  = Path(args.left_dir)  if args.left_dir  else Path(args.job_dir) / 'left'
    right_dir = Path(args.right_dir) if args.right_dir else Path(args.job_dir) / 'right'

    job_dir    = left_dir.parent  # kept for any legacy references
    output_dir = Path(args.output_dir)

    # ── Load stage bounds and inter-stage transform ───────────────────────────
    print(f"Loading stage info from {args.poses_world}...")
    stage1_fids, stage2_fids, T_obj_stage1, T_obj_stage2 = load_stage_fid_sets(Path(args.poses_world))
    valid_fid_set = stage1_fids | stage2_fids
    print(f"  Stage 1: {len(stage1_fids)} frames, Stage 2: {len(stage2_fids)} frames")

    # ── Build timestamp → seq_idx index ──────────────────────────────────────
    print("Building timestamp → seq_idx index from frames_meta.json...")
    ts_us_to_seq_idx = build_timestamp_to_seq_idx(Path(args.frames_meta), camera=args.camera)
    print(f"  {len(ts_us_to_seq_idx)} frames indexed")

    # ── Load SfM keyframes ────────────────────────────────────────────────────
    print(f"Loading SfM keyframes from {args.sfm_keyframes}...")
    keyframe_list = load_sfm_keyframes(
        Path(args.sfm_keyframes), ts_us_to_seq_idx, valid_fid_set, args.camera)
    n_s1 = sum(1 for fid, _ in keyframe_list if fid in stage1_fids)
    n_s2 = len(keyframe_list) - n_s1
    print(f"  {len(keyframe_list)} keyframes selected ({n_s1} stage1, {n_s2} stage2)")

    # ── Create output structure ───────────────────────────────────────────────
    for subdir in ('left', 'right', 'poses', 'masks', 'depth'):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # ── Collect valid frames ──────────────────────────────────────────────────
    print("Collecting frames...")
    frames = []   # (out_idx, seq_idx, cam_in_ob, img_src, img_src_right_or_None)
    skipped = 0

    for out_idx, (seq_idx, T_world_from_cam) in enumerate(keyframe_list):
        img_src = left_dir / f'{seq_idx:06d}.jpg'
        if not img_src.exists():
            skipped += 1
            continue

        img_src_right = right_dir / f'{seq_idx:06d}.jpg'
        if not img_src_right.exists():
            img_src_right = None

        # cam_in_ob = inv(T_world_from_obj_stage) @ T_cam_world
        # This expresses the camera in the object's canonical frame for each stage.
        # Stage 1 and stage 2 cameras all view the same object in the same object frame.
        if seq_idx in stage2_fids:
            cam_in_ob = np.linalg.inv(T_obj_stage2) @ T_world_from_cam
        else:
            cam_in_ob = np.linalg.inv(T_obj_stage1) @ T_world_from_cam

        frames.append((out_idx, seq_idx, cam_in_ob, img_src, img_src_right))

    # ── Write poses and keyframes.yml ─────────────────────────────────────────
    keyframes = {}
    for out_idx, fid, cam_in_ob, img_src, img_src_right in frames:
        out_stem = f'left{out_idx:06d}'
        with open(output_dir / 'poses' / f'{out_stem}.json', 'w') as pf:
            json.dump(cam_in_ob.tolist(), pf)
        keyframes[f'keyframe_{out_stem}'] = {'cam_in_ob': cam_in_ob.flatten().tolist()}

    # ── Convert images to PNG in parallel ────────────────────────────────────
    def convert_frame(item):
        out_idx, fid, T, img_src, img_src_right = item
        save_as_png(img_src,       output_dir / 'left'  / f'left{out_idx:06d}.png')
        if img_src_right:
            save_as_png(img_src_right, output_dir / 'right' / f'right{out_idx:06d}.png')

    n_total   = len(frames)
    n_workers = min(os.cpu_count(), 16)
    print(f"Converting {n_total} frames to PNG (workers={n_workers})...")
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(convert_frame, f): f for f in frames}
        for fut in as_completed(futures):
            fut.result()
            completed += 1
            if completed % 100 == 0:
                print(f"  {completed}/{n_total} frames")

    # ── Write keyframes.yml ───────────────────────────────────────────────────
    with open(output_dir / 'keyframes.yml', 'w') as f:
        yaml.dump(keyframes, f, default_flow_style=None, sort_keys=False)

    # ── Depth symlinks ────────────────────────────────────────────────────────
    if args.depth_dir:
        depth_src_dir  = Path(args.depth_dir)
        depth_link_dir = output_dir / 'depth'
        depth_link_dir.mkdir(exist_ok=True)
        for out_idx, seq_idx, _, _, _ in frames:
            src = depth_src_dir / f'{seq_idx:06d}.png'
            dst = depth_link_dir / f'left{out_idx:06d}.png'
            try:
                dst.unlink()
            except FileNotFoundError:
                pass
            dst.symlink_to(os.path.relpath(src, dst.parent))
        print(f"  Depth symlinks: {len(frames)}")

    # ── Mask symlinks ─────────────────────────────────────────────────────────
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
    print(f"  Frames written: {len(keyframes)}  (skipped: {skipped})")
    print(f"  Output: {output_dir}")
    print(f"  keyframes.yml: {len(keyframes)} entries")


if __name__ == '__main__':
    main()
