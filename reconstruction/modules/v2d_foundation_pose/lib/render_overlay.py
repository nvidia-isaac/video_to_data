# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Render mesh overlay on video frames using per-frame object-to-camera poses.

Inputs:
  --video_path              path to input video
  --poses_dir               directory of per-frame 4×4 matrix JSONs (T_cam_from_obj)
  --mesh_path               path to mesh .obj file
  --camera_intrinsics_path  JSON with {fx, fy, cx, cy, width, height}
  --output_dir              output directory for overlay frames (%06d.jpg)
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh

# CV camera → OpenGL camera: flip Y and Z axes
_T_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def render_overlay(
    video_path: str,
    poses_dir: str,
    mesh_path: str,
    camera_intrinsics_path: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(camera_intrinsics_path) as f:
        intr = json.load(f)
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    width, height = int(intr["width"]), int(intr["height"])

    tm = trimesh.load(mesh_path, force="mesh")
    mesh = pyrender.Mesh.from_trimesh(tm)

    poses = {}
    for pf in sorted(Path(poses_dir).glob("*.json")):
        d = json.loads(pf.read_text())
        if isinstance(d, list):
            poses[int(pf.stem)] = np.array(d)

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    renderer = pyrender.OffscreenRenderer(width, height)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in poses:
            T_cam_from_obj = poses[frame_idx]
            scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.4, 0.4, 0.4])
            scene.add(mesh, pose=T_cam_from_obj)
            scene.add(camera, pose=_T_CV_TO_GL)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            scene.add(light, pose=_T_CV_TO_GL)

            color, depth = renderer.render(scene)
            mask = (depth > 0).astype(np.float32)[:, :, np.newaxis]
            overlay_bgr = color[:, :, ::-1]  # RGB to BGR
            out = np.clip(mask * overlay_bgr + (1.0 - mask) * frame, 0, 255).astype(np.uint8)
        else:
            out = frame

        cv2.imwrite(os.path.join(output_dir, f"{frame_idx:06d}.jpg"), out)
        frame_idx += 1

    cap.release()
    renderer.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render mesh overlay on video frames")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--poses_dir", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--camera_intrinsics_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    render_overlay(
        args.video_path,
        args.poses_dir,
        args.mesh_path,
        args.camera_intrinsics_path,
        args.output_dir,
    )
