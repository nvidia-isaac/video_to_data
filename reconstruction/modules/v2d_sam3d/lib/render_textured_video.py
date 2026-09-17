# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a textured SRT mesh overlay video using CuSFM camera poses.

For each eligible SfM keyframe the mesh is rendered offscreen and alpha-
composited onto the source image. A frame-end bound supports two-stage scans;
without a bound, all keyframes are rendered for a stationary-object capture.
Uses pyrender's PyOpenGL EGL backend
(EGL_EXT_platform_device), which works with NVIDIA GPU containers that do not
expose a DRM render device.

An earlier version used open3d.visualization.rendering.OffscreenRenderer
(Filament + GBM EGL), but Filament's EGL backend needs /dev/dri/renderD128.
OSMO GPU pods do not expose that device, so eglInitialize fails and Filament
segfaults on the null display.

Usage (inside container):
    python -m v2d.sam3d.lib.render_textured_video \
        --job_dir  /data/job \
        --glb_path /data/job/sam3d/000651/srt/output_scaled.glb \
        --output_dir /data/job/sam3d/000651/render_video_frames \
        --frame_end 319
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

# Must be set before pyrender / OpenGL imports.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh


# ── SfM pose loading ──────────────────────────────────────────────────────────

def _aa_to_matrix(aa: dict) -> np.ndarray:
    x, y, z = aa["x"], aa["y"], aa["z"]
    angle = math.radians(aa["angle_degrees"])
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = x / norm, y / norm, z / norm
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


def _load_sfm_poses(job_dir: Path) -> dict[str, np.ndarray]:
    """Return {frame_id: T_cam_from_world (4×4, OpenCV)} for left-camera keyframes."""
    with open(job_dir / "frames_meta.json") as f:
        meta = json.load(f)
    cam_params = meta["camera_params_id_to_camera_params"]

    left_sids: dict[int, int] = {}
    right_sids: set[int] = set()
    for kf in meta["keyframes_metadata"]:
        sid = int(kf["synced_sample_id"])
        sensor = cam_params[kf["camera_params_id"]]["sensor_meta_data"]["sensor_name"]
        if "front_stereo_camera_left" in sensor:
            left_sids[sid] = int(kf["timestamp_microseconds"])
        elif "front_stereo_camera_right" in sensor:
            right_sids.add(sid)

    common = sorted(set(left_sids) & right_sids)
    ts_to_seq = {left_sids[s]: i for i, s in enumerate(common)}

    with open(job_dir / "sfm" / "keyframes" / "frames_meta.json") as f:
        sfm = json.load(f)

    poses: dict[str, np.ndarray] = {}
    for kf in sfm["keyframes_metadata"]:
        if "front_stereo_camera_left" not in kf.get("image_name", ""):
            continue
        seq_idx = ts_to_seq.get(int(kf["timestamp_microseconds"]))
        if seq_idx is None:
            continue
        aa = kf["camera_to_world"]["axis_angle"]
        t  = kf["camera_to_world"]["translation"]
        T_c2w = np.eye(4)
        T_c2w[:3, :3] = _aa_to_matrix(aa)
        T_c2w[:3,  3] = [t["x"], t["y"], t["z"]]
        poses[f"{seq_idx:06d}"] = np.linalg.inv(T_c2w)
    return poses


# ── Mesh loading ──────────────────────────────────────────────────────────────

def _load_glb_meshes(glb_path: Path) -> list[trimesh.Trimesh]:
    """Load GLB geometries with scene-graph transforms baked in."""
    loaded = trimesh.load(str(glb_path), process=False)
    if isinstance(loaded, trimesh.Scene):
        return list(loaded.dump())
    return [loaded]


# OpenCV camera frame (x-right, y-down, z-forward) → OpenGL camera frame
# (x-right, y-up, z-back) which pyrender expects. Composing on the right of
# T_world_from_cam_cv re-expresses the camera axes.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


# ── Entry point ───────────────────────────────────────────────────────────────

def render_textured_video(
    job_dir: Path,
    glb_path: Path,
    output_dir: Path,
    frame_end: int | None = None,
    alpha: float = 0.6,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Prevent stale frames from a previous capture mode or shorter rerun from
    # being included when the directory is stitched into a video.
    for old_frame in output_dir.glob("*.jpg"):
        old_frame.unlink()

    # Intrinsics
    intr_files = sorted((job_dir / "intrinsics").glob("*.json"))
    if not intr_files:
        raise FileNotFoundError(f"No intrinsics in {job_dir / 'intrinsics'}")
    with open(intr_files[0]) as f:
        intr = json.load(f)
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    width, height = int(intr["width"]), int(intr["height"])

    # A two-stage caller supplies the Stage-1 end; a stationary capture uses all.
    poses = _load_sfm_poses(job_dir)
    frame_ids = sorted(
        [n for n in poses if frame_end is None or int(n) <= frame_end],
        key=lambda n: int(n),
    )
    if not frame_ids:
        suffix = "" if frame_end is None else f" <= frame_end={frame_end}"
        raise ValueError(f"No eligible SfM keyframes{suffix}")
    frame_scope = "all keyframes" if frame_end is None else f"keyframes <= {frame_end}"
    print(f"[render_textured] {len(frame_ids)} {frame_scope}")

    # Mesh
    print(f"[render_textured] loading mesh: {glb_path}")
    meshes = _load_glb_meshes(glb_path)
    n_verts = sum(len(mesh.vertices) for mesh in meshes)
    n_faces = sum(len(mesh.faces) for mesh in meshes)
    print(
        f"[render_textured] mesh: {n_verts} vertices, {n_faces} faces "
        f"across {len(meshes)} geometr{'y' if len(meshes) == 1 else 'ies'}"
    )

    # Build the scene once and reuse it for every camera pose.
    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        ambient_light=np.array([0.3, 0.3, 0.3]),
    )
    for mesh in meshes:
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    cam_node = scene.add(camera, pose=np.eye(4))

    # The light follows the camera so the visible object surface remains lit.
    light = pyrender.DirectionalLight(
        color=np.array([1.0, 1.0, 1.0]),
        intensity=3.0,
    )
    light_node = scene.add(light, pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(width, height)

    images_dir = job_dir / "left"
    n_written  = 0
    n_total    = len(frame_ids)
    try:
        for frame_id in frame_ids:
            img_path = images_dir / f"{frame_id}.jpg"
            if not img_path.exists():
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            # SfM gives T_cam_from_world (OpenCV). Pyrender wants
            # T_world_from_cam in OpenGL convention.
            T_world_from_cam_cv = np.linalg.inv(poses[frame_id])
            T_world_from_cam_gl = T_world_from_cam_cv @ _CV_TO_GL
            scene.set_pose(cam_node, T_world_from_cam_gl)
            scene.set_pose(light_node, T_world_from_cam_gl)

            color, depth = renderer.render(scene)
            mask = depth > 0.0

            color_bgr = color[:, :, ::-1].astype(np.float32)
            img_f = img_bgr.astype(np.float32)
            blended = (
                color_bgr * alpha + img_f * (1.0 - alpha)
            ).clip(0, 255).astype(np.uint8)
            result = np.where(mask[:, :, None], blended, img_bgr)

            out_path = output_dir / f"{n_written:06d}.jpg"
            cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 90])
            n_written += 1
            print(
                f"[render_textured] frame {n_written}/{n_total}  (id={frame_id})",
                flush=True,
            )
    finally:
        renderer.delete()

    print(f"[render_textured] wrote {n_written} frames → {output_dir}")
    return n_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_dir",          required=True)
    parser.add_argument("--glb_path",         required=True)
    parser.add_argument("--output_dir",       required=True)
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument(
        "--frame_end",
        type=int,
        default=None,
        help="Inclusive last frame to render; omit to render all keyframes",
    )
    frame_group.add_argument(
        "--stage1_end_frame",
        dest="frame_end",
        type=int,
        help="Deprecated alias for --frame_end",
    )
    parser.add_argument("--alpha",            type=float, default=0.6)
    args = parser.parse_args()

    render_textured_video(
        job_dir=Path(args.job_dir),
        glb_path=Path(args.glb_path),
        output_dir=Path(args.output_dir),
        frame_end=args.frame_end,
        alpha=args.alpha,
    )
