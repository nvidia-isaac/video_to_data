"""Render a Stage-1 overlay video: textured SRT mesh projected via pyrender.

For each Stage-1 SfM keyframe the mesh is rendered offscreen and alpha-
composited onto the source image. Uses pyrender's PyOpenGL EGL backend
(EGL_EXT_platform_device) — the same backend used by every other GPU
renderer in this repo (v2d_nlf, v2d_foundation_pose, v2d_mesh, v2d_mv,
v2d_sam3d_body).

An earlier version used open3d.visualization.rendering.OffscreenRenderer
(Filament + GBM EGL), but Filament's EGL backend needs /dev/dri/renderD128
which OSMO GPU pods don't expose, so eglInitialize fails and Filament
segfaults on the NULL display.

Usage (inside container):
    python -m v2d.sam3d.lib.render_textured_video \\
        --job_dir  /data/job \\
        --glb_path /data/job/sam3d/000651/srt/output_scaled.glb \\
        --output_dir /data/job/sam3d/000651/render_video_frames \\
        --stage1_end_frame 319
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
    """Load all geometries from a GLB with their scene-graph transforms baked in.

    SRT GLBs carry a node hierarchy; trimesh.Scene.dump() walks the graph and
    applies each node's transform. networkx is required for this and is
    already pinned in v2d_sam3d/docker/Dockerfile.
    """
    loaded = trimesh.load(str(glb_path), process=False)
    if isinstance(loaded, trimesh.Scene):
        return list(loaded.dump())
    return [loaded]


# ── Entry point ───────────────────────────────────────────────────────────────

# OpenCV camera frame (x-right, y-down, z-forward) → OpenGL camera frame
# (x-right, y-up, z-back) which pyrender expects. Composing on the right of
# T_world_from_cam_cv re-expresses the cam axes.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def render_textured_video(
    job_dir: Path,
    glb_path: Path,
    output_dir: Path,
    stage1_end_frame: int,
    alpha: float = 0.6,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intrinsics
    intr_files = sorted((job_dir / "intrinsics").glob("*.json"))
    if not intr_files:
        raise FileNotFoundError(f"No intrinsics in {job_dir / 'intrinsics'}")
    with open(intr_files[0]) as f:
        intr = json.load(f)
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    width, height = int(intr["width"]), int(intr["height"])

    # SfM poses filtered to Stage-1
    poses = _load_sfm_poses(job_dir)
    stage1_frames = sorted(
        [n for n in poses if int(n) <= stage1_end_frame],
        key=lambda n: int(n),
    )
    if not stage1_frames:
        raise ValueError(f"No SfM keyframes ≤ stage1_end_frame={stage1_end_frame}")
    print(f"[render_textured] {len(stage1_frames)} Stage-1 keyframes (≤ {stage1_end_frame})")

    # Mesh
    print(f"[render_textured] loading mesh: {glb_path}")
    meshes = _load_glb_meshes(glb_path)
    n_verts = sum(len(m.vertices) for m in meshes)
    n_faces = sum(len(m.faces) for m in meshes)
    print(f"[render_textured] mesh: {n_verts} vertices, {n_faces} faces "
          f"across {len(meshes)} geometr{'y' if len(meshes) == 1 else 'ies'}")

    # Build scene once, reuse for all frames.
    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        ambient_light=np.array([0.3, 0.3, 0.3]),
    )
    for m in meshes:
        scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    cam_node = scene.add(camera, pose=np.eye(4))

    # Headlight: light follows the camera so the mesh is always lit from the
    # viewer's side, regardless of which keyframe we're rendering.
    light = pyrender.DirectionalLight(color=np.array([1.0, 1.0, 1.0]),
                                      intensity=3.0)
    light_node = scene.add(light, pose=np.eye(4))

    renderer = pyrender.OffscreenRenderer(width, height)

    images_dir = job_dir / "left"
    n_written  = 0
    n_total    = len(stage1_frames)
    try:
        for frame_id in stage1_frames:
            img_path = images_dir / f"{frame_id}.jpg"
            if not img_path.exists():
                continue
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            # SfM gives T_cam_from_world (OpenCV). pyrender wants
            # T_world_from_cam in OpenGL convention.
            T_world_from_cam_cv = np.linalg.inv(poses[frame_id])
            T_world_from_cam_gl = T_world_from_cam_cv @ _CV_TO_GL
            scene.set_pose(cam_node,   T_world_from_cam_gl)
            scene.set_pose(light_node, T_world_from_cam_gl)

            color, depth = renderer.render(scene)        # color: H×W×3 RGB uint8
            mask = depth > 0.0

            color_bgr = color[:, :, ::-1].astype(np.float32)
            img_f     = img_bgr.astype(np.float32)
            blended   = (color_bgr * alpha + img_f * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
            result    = np.where(mask[:, :, None], blended, img_bgr)

            out_path = output_dir / f"{n_written:06d}.jpg"
            cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 90])
            n_written += 1
            print(f"[render_textured] frame {n_written}/{n_total}  (id={frame_id})", flush=True)
    finally:
        renderer.delete()

    print(f"[render_textured] wrote {n_written} frames → {output_dir}")
    return n_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_dir",          required=True)
    parser.add_argument("--glb_path",         required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--stage1_end_frame", type=int, required=True)
    parser.add_argument("--alpha",            type=float, default=0.6)
    args = parser.parse_args()

    render_textured_video(
        job_dir=Path(args.job_dir),
        glb_path=Path(args.glb_path),
        output_dir=Path(args.output_dir),
        stage1_end_frame=args.stage1_end_frame,
        alpha=args.alpha,
    )
