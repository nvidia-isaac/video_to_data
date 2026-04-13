"""Render a Stage-1 overlay video: textured SRT mesh projected via open3d.

Runs inside the v2d_sam3d container (has open3d with Filament/EGL backend).
For each Stage-1 SfM keyframe the mesh is rendered offscreen with its original
texture and alpha-composited onto the source image.

Usage (inside container):
    python -m v2d.sam3d.lib.render_textured_video \
        --job_dir  /data/job \
        --glb_path /data/job/sam3d/000651/srt/output_scaled.glb \
        --output_dir /data/job/sam3d/000651/render_video_frames \
        --stage1_end_frame 319
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


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


# ── Entry point ───────────────────────────────────────────────────────────────

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
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]])

    # SfM poses filtered to Stage-1
    poses = _load_sfm_poses(job_dir)
    stage1_frames = sorted(
        [n for n in poses if int(n) <= stage1_end_frame],
        key=lambda n: int(n),
    )
    if not stage1_frames:
        raise ValueError(f"No SfM keyframes ≤ stage1_end_frame={stage1_end_frame}")
    print(f"[render_textured] {len(stage1_frames)} Stage-1 keyframes (≤ {stage1_end_frame})")

    # Load mesh
    print(f"[render_textured] loading mesh: {glb_path}")
    mesh = o3d.io.read_triangle_mesh(str(glb_path), enable_post_processing=True)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    print(f"[render_textured] mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles, "
          f"has_textures={mesh.has_textures()}, has_vertex_colors={mesh.has_vertex_colors()}")

    # Build material: prefer texture if available
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"

    # Build renderer once, reuse for all frames
    render = rendering.OffscreenRenderer(width, height)
    render.scene.set_background(np.array([0.0, 0.0, 0.0, 1.0]))
    render.scene.add_geometry("mesh", mesh, mat)
    render.scene.scene.set_sun_light(
        direction=[-1.0, -1.0, -1.0],
        color=[1.0, 1.0, 1.0],
        intensity=75000,
    )
    render.scene.scene.enable_sun_light(True)

    images_dir = job_dir / "left"
    n_written  = 0
    n_total    = len(stage1_frames)
    for frame_id in stage1_frames:
        img_path = images_dir / f"{frame_id}.jpg"
        if not img_path.exists():
            continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        # Set camera: K (3×3) + T_cam_from_world (4×4, OpenCV convention)
        render.setup_camera(K, poses[frame_id], width, height)

        color_o3d = render.render_to_image()          # RGB uint8
        depth_o3d = render.render_to_depth_image()    # float32 metres

        color_rgb = np.asarray(color_o3d)             # H×W×3
        depth     = np.asarray(depth_o3d)             # H×W

        # Composite: blend where mesh is visible (depth > 0 and < far plane)
        mask       = (depth > 0.0) & (depth < 1e4)
        color_bgr  = color_rgb[:, :, ::-1].astype(np.float32)
        img_f      = img_bgr.astype(np.float32)
        blended    = (color_bgr * alpha + img_f * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        result     = np.where(mask[:, :, None], blended, img_bgr)

        out_path = output_dir / f"{n_written:06d}.jpg"
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, 90])
        n_written += 1
        print(f"[render_textured] frame {n_written}/{n_total}  (id={frame_id})", flush=True)

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
