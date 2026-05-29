# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Verification: render projected MANO meshes onto the source video.

Accepts either the per-frame layout produced by ``video_to_hands`` /
``image_list_to_hands`` (``<wilor_dir>/<frame:06d>.json`` containing a list of
detections) or the per-track layout produced by ``masks_to_hands``
(``<wilor_dir>/<track_id>/<frame:06d>.json`` containing a single record). The
mode is auto-detected from the directory contents.

Usage:
    python -m v2d.wilor.lib.render_hands_video \\
        --frames_dir /data/frames \\
        --wilor_dir  /data/wilor \\
        --mano_assets_root /data/weights/wilor/mano \\
        --output_path /data/wilor_overlay.mp4
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tempfile
from typing import Tuple

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import torch
import trimesh
from manotorch.manolayer import ManoLayer
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


_COLORS = [
    (255,  60,  60), ( 60, 160, 255), ( 60, 200,  60),
    (255, 180,  40), (200,  80, 220),
]
_CV_TO_GL = np.array([1.0, -1.0, -1.0])


def _font(size: int = 18):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _mano_layer(mano_assets_root: str) -> ManoLayer:
    return ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="right",
        center_idx=None,
        mano_assets_root=mano_assets_root,
    )


def _frame_records(wilor_dir: str) -> dict[int, list[dict]]:
    """Map frame_idx → list of records. Auto-detects per-frame vs per-track layout."""
    # Per-track layout: <track>/<frame>.json
    track_files = sorted(glob.glob(os.path.join(wilor_dir, "*", "*.json")))
    if track_files:
        out: dict[int, list[dict]] = {}
        for path in track_files:
            with open(path) as f:
                rec = json.load(f)
            out.setdefault(int(rec["frame_idx"]), []).append(rec)
        return out

    # Per-frame layout: <frame>.json (list)
    frame_files = sorted(glob.glob(os.path.join(wilor_dir, "*.json")))
    out = {}
    for path in frame_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            f_idx = int(stem)
        except ValueError:
            continue
        with open(path) as f:
            recs = json.load(f)
        # Synthesize a track_id from the index so the legend stays useful.
        for i, rec in enumerate(recs):
            rec.setdefault("frame_idx", f_idx)
            rec.setdefault("track_id", i + 1)
        out[f_idx] = recs
    return out


def _build_mesh_for_record(rec: dict, mano: ManoLayer) -> Tuple[np.ndarray, np.ndarray]:
    mano_p = rec["mano"]
    pose_aa = np.concatenate([
        np.array(mano_p["global_orient"], dtype=np.float32),
        np.array(mano_p["hand_pose"],     dtype=np.float32),
    ])
    betas_aa = np.array(mano_p["betas"], dtype=np.float32)
    out = mano(
        torch.from_numpy(pose_aa)[None],
        torch.from_numpy(betas_aa)[None],
    )
    verts_local = out.verts[0].detach().numpy()
    if not rec["is_right"]:
        verts_local[:, 0] *= -1
    cam_t = np.array(rec["camera"]["pred_cam_t_full"], dtype=np.float64)
    verts_cam = verts_local + cam_t[None, :]
    faces = mano.th_faces.numpy()
    if not rec["is_right"]:
        faces = faces[:, [0, 2, 1]]
    return verts_cam.astype(np.float32), faces


def render_hands_video(
    frames_dir: str,
    wilor_dir: str,
    mano_assets_root: str,
    output_path: str,
    fps: float = 30.0,
    alpha: float = 0.55,
) -> None:
    frame_files = sorted(
        glob.glob(os.path.join(frames_dir, "*.png")) +
        glob.glob(os.path.join(frames_dir, "*.jpg"))
    )
    if not frame_files:
        raise FileNotFoundError(f"No frames in {frames_dir}")

    records = _frame_records(wilor_dir)
    if not records:
        raise FileNotFoundError(f"No WiLoR JSONs in {wilor_dir}")

    mano = _mano_layer(mano_assets_root)

    W, H = Image.open(frame_files[0]).size
    first_rec = next(iter(records.values()))[0]
    focal = first_rec["camera"]["scaled_focal_length"]
    z_max = max(rec["camera"]["pred_cam_t_full"][2]
                for recs in records.values() for rec in recs)
    zfar = max(50.0, float(z_max) * 1.5)
    cam = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=W / 2, cy=H / 2,
                                    znear=0.01, zfar=zfar)
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    font = _font(18)

    with tempfile.TemporaryDirectory() as tmpdir:
        for f_idx, frame_path in enumerate(tqdm(frame_files, desc="render", ncols=80)):
            bg = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32)
            recs = records.get(f_idx, [])
            if not recs:
                Image.fromarray(bg.astype(np.uint8)).save(
                    os.path.join(tmpdir, f"{f_idx:06d}.png"))
                continue

            scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5])
            scene.add(cam, pose=np.eye(4))
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0),
                      pose=np.eye(4))
            for rec in recs:
                verts_cv, faces = _build_mesh_for_record(rec, mano)
                verts_gl = verts_cv * _CV_TO_GL
                color = _COLORS[(rec.get("track_id", 1) - 1) % len(_COLORS)]
                tm = trimesh.Trimesh(verts_gl, faces, process=False)
                tm.visual.face_colors = np.array([*color, 255], dtype=np.uint8)
                scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))
            rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            ov = rgba.astype(np.float32)
            mask = ov[:, :, 3:4] / 255.0
            out = bg * (1 - alpha * mask) + ov[:, :, :3] * (alpha * mask)
            out_img = Image.fromarray(out.clip(0, 255).astype(np.uint8))

            draw = ImageDraw.Draw(out_img)
            for i, rec in enumerate(recs):
                color = _COLORS[(rec.get("track_id", 1) - 1) % len(_COLORS)]
                y = 8 + i * 24
                draw.rectangle([8, y, 28, y + 20], fill=color)
                draw.text((34, y),
                          f"id={rec.get('track_id', i + 1)}  {'R' if rec['is_right'] else 'L'}",
                          fill=(255, 255, 255), font=font,
                          stroke_width=2, stroke_fill=(0, 0, 0))
            out_img.save(os.path.join(tmpdir, f"{f_idx:06d}.png"))

        renderer.delete()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-r", str(fps),
            "-i", os.path.join(tmpdir, "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            output_path,
        ], check=True)
    print(f"Saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir",       required=True)
    parser.add_argument("--wilor_dir",        required=True)
    parser.add_argument("--mano_assets_root", required=True,
                        help="Dir containing MANO_RIGHT.pkl (manotorch convention).")
    parser.add_argument("--output_path",      required=True)
    parser.add_argument("--fps",   type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    args = parser.parse_args()
    render_hands_video(
        frames_dir       = args.frames_dir,
        wilor_dir        = args.wilor_dir,
        mano_assets_root = args.mano_assets_root,
        output_path      = args.output_path,
        fps              = args.fps,
        alpha            = args.alpha,
    )


if __name__ == "__main__":
    main()
