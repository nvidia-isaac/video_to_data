# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Prepare a FoundationPose job directory from stereo image data.

Input data format (e.g. mapping_data/<session>/):
  frames_meta.json                  - camera params + keyframe metadata
  front_stereo_camera_left/*.jpeg   - left camera images (timestamp filenames)
  front_stereo_camera_right/*.jpeg  - right camera images (timestamp filenames)

Output job directory structure:
  <job_dir>/
    left/          000000.jpg, 000001.jpg, ...   (sorted left frames)
    right/         000000.jpg, 000001.jpg, ...   (matched right frames)
    intrinsics/    000000.json, ...              (rectified camera intrinsics per frame)
    calibration.json                             (stereo calibration for FoundationStereo)
    video.mp4                                    (left camera video for SAM2)

Usage:
  python3 prepare_job.py \\
      --input_dir /path/to/session \\
      --job_dir reconstruction/data/my_job \\
      [--fps 30] \\
      [--max_frames N]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def load_frames_meta(input_dir):
    meta_path = os.path.join(input_dir, "frames_meta.json")
    with open(meta_path) as f:
        return json.load(f)


def get_rectified_intrinsics(camera_params):
    """Extract rectified intrinsics from the projection matrix (3x4)."""
    proj = camera_params["calibration_parameters"]["projection_matrix"]["data"]
    # Projection matrix row-major: [fx, 0, cx, tx,  0, fy, cy, 0,  0, 0, 1, 0]
    fx = proj[0]
    fy = proj[5]
    cx = proj[2]
    cy = proj[6]
    width = camera_params["calibration_parameters"]["image_width"]
    height = camera_params["calibration_parameters"]["image_height"]
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": width, "height": height}


def build_stereo_pairs(meta):
    """
    Return sorted list of (synced_sample_id, left_image_name, right_image_name).
    Pairs left and right frames by synced_sample_id.
    """
    left_frames = {}   # synced_sample_id -> image_name
    right_frames = {}  # synced_sample_id -> image_name

    cam_params = meta["camera_params_id_to_camera_params"]

    for kf in meta["keyframes_metadata"]:
        cam_id = kf["camera_params_id"]
        sid = int(kf["synced_sample_id"])
        img = kf["image_name"]
        sensor_name = cam_params[cam_id]["sensor_meta_data"]["sensor_name"]
        if "left" in sensor_name:
            left_frames[sid] = img
        elif "right" in sensor_name:
            right_frames[sid] = img

    # Only keep synced_sample_ids that have both left and right
    common_ids = sorted(set(left_frames) & set(right_frames))
    return [(sid, left_frames[sid], right_frames[sid]) for sid in common_ids]


def prepare_job(input_dir, job_dir, fps=30, max_frames=None):
    meta = load_frames_meta(input_dir)
    cam_params = meta["camera_params_id_to_camera_params"]

    # Get rectified intrinsics from left camera projection matrix
    left_cam_id = meta["stereo_pair"][0]["left_camera_param_id"]
    right_cam_id = meta["stereo_pair"][0]["right_camera_param_id"]
    intrinsics = get_rectified_intrinsics(cam_params[left_cam_id])
    baseline = meta["stereo_pair"][0]["baseline_meters"]

    # Build matched stereo pairs
    pairs = build_stereo_pairs(meta)
    if max_frames:
        pairs = pairs[:max_frames]

    print(f"Found {len(pairs)} matched stereo pairs")

    # Create output directories
    left_dir = os.path.join(job_dir, "left")
    right_dir = os.path.join(job_dir, "right")
    intrinsics_dir = os.path.join(job_dir, "intrinsics")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    os.makedirs(intrinsics_dir, exist_ok=True)

    # Copy images with sequential names
    print("Copying images...")
    for idx, (sid, left_img, right_img) in enumerate(pairs):
        src_left = os.path.abspath(os.path.join(input_dir, left_img))
        src_right = os.path.abspath(os.path.join(input_dir, right_img))
        dst_left = os.path.join(left_dir, f"{idx:06d}.jpg")
        dst_right = os.path.join(right_dir, f"{idx:06d}.jpg")

        if not os.path.exists(dst_left):
            shutil.copy2(src_left, dst_left)
        if not os.path.exists(dst_right):
            shutil.copy2(src_right, dst_right)

        # Write per-frame intrinsics JSON
        intrinsics_path = os.path.join(intrinsics_dir, f"{idx:06d}.json")
        if not os.path.exists(intrinsics_path):
            with open(intrinsics_path, "w") as f:
                json.dump(intrinsics, f, indent=4)

        if idx % 100 == 0 and idx > 0:
            print(f"  {idx}/{len(pairs)}")

    # Write stereo calibration JSON (for FoundationStereo)
    calibration = {**intrinsics, "baseline": baseline}
    calibration_path = os.path.join(job_dir, "calibration.json")
    with open(calibration_path, "w") as f:
        json.dump(calibration, f, indent=4)
    print(f"Wrote calibration: {calibration_path}")

    # Create video.mp4 from left images (for SAM2)
    video_path = os.path.join(job_dir, "video.mp4")
    if not os.path.exists(video_path):
        print("Creating video.mp4 from left images...")
        # Prefer libx264 (software); fall back to h264_nvenc (NVIDIA GPU) or mpeg4 if unavailable
        for codec in ["libx264", "h264_nvenc", "mpeg4"]:
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(left_dir, "%06d.jpg"),
                "-c:v", codec,
                "-pix_fmt", "yuv420p",
                video_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                break
            reason = result.stderr.strip().splitlines()
            print(f"  codec {codec} failed{': ' + reason[-1][:100] if reason else ''}, trying next...")
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            sys.exit(1)
        print(f"Wrote video: {video_path}")
    else:
        print(f"video.mp4 already exists, skipping")

    # Copy frames_meta.json so Docker containers can access it via /data/ mount
    frames_meta_dst = os.path.join(job_dir, "frames_meta.json")
    if not os.path.exists(frames_meta_dst):
        shutil.copy2(os.path.join(input_dir, "frames_meta.json"), frames_meta_dst)
        print(f"Copied frames_meta.json to {frames_meta_dst}")

    print(f"\nJob directory ready: {job_dir}")
    print(f"  Frames:      {len(pairs)}")
    print(f"  Resolution:  {intrinsics['width']}x{intrinsics['height']}")
    print(f"  Intrinsics:  fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")
    print(f"  Baseline:    {baseline:.4f} m")
    print(f"\nNext steps:")
    print(f"  1. Run SAM2 to generate masks:  docker compose run sam2-annotate  (annotate object in first frame)")
    print(f"  2. Run FoundationStereo for depth: docker compose run foundation-stereo-image-list-to-depth ...")
    print(f"  3. Run SAM3D for mesh:           docker compose run sam3d-image-to-mesh ...")
    print(f"  4. Run FoundationPose:           docker compose run foundationpose-video-to-poses ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FoundationPose job from stereo image data")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to session data directory")
    parser.add_argument("--job_dir", type=str, required=True, help="Output job directory")
    parser.add_argument("--fps", type=int, default=30, help="FPS for output video (default: 30)")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit number of frames (optional)")
    args = parser.parse_args()

    prepare_job(args.input_dir, args.job_dir, fps=args.fps, max_frames=args.max_frames)
