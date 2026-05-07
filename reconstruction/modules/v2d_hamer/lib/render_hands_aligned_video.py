"""Render projected MANO meshes onto the source video — aligned schema.

Sibling of ``render_hands_video.py``, but consumes the JSONs written by
``align_hands.py`` (real intrinsics, depth-shifted cam_t) instead of HaMeR's
virtual-pinhole output. The intrinsics come from each JSON; the renderer
uses them directly.

Usage:
    python -m v2d.hamer.lib.render_hands_aligned_video \\
        --frames_dir /data/frames \\
        --aligned_dir /data/hamer_aligned \\
        --mano_assets_root /data/weights/hamer/_DATA/data \\
        --output_path /data/hamer_aligned_overlay.mp4
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


_TRACK_COLORS = [
    (255,  60,  60),   # red
    ( 60, 160, 255),   # blue
    ( 60, 200,  60),   # green
    (255, 180,  40),   # orange
    (200,  80, 220),   # purple
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


def _frame_records(aligned_dir: str) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(aligned_dir, "*", "*.json"))):
        with open(path) as f:
            rec = json.load(f)
        out.setdefault(int(rec["frame_idx"]), []).append(rec)
    return out


def _build_mesh(rec: dict, mano: ManoLayer) -> Tuple[np.ndarray, np.ndarray]:
    pose_aa = np.concatenate([
        np.array(rec["mano"]["global_orient"], dtype=np.float32),
        np.array(rec["mano"]["hand_pose"],     dtype=np.float32),
    ])
    betas = np.array(rec["mano"]["betas"], dtype=np.float32)
    out = mano(torch.from_numpy(pose_aa)[None], torch.from_numpy(betas)[None])
    verts_local = out.verts[0].detach().numpy()
    if not rec["is_right"]:
        verts_local[:, 0] *= -1
    cam_t = np.array(rec["cam_t"], dtype=np.float64)
    verts_cam = verts_local + cam_t[None, :]
    faces = mano.th_faces.numpy()
    if not rec["is_right"]:
        faces = faces[:, [0, 2, 1]]
    return verts_cam.astype(np.float32), faces


def render_hands_aligned_video(
    frames_dir: str,
    aligned_dir: str,
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

    records = _frame_records(aligned_dir)
    if not records:
        raise FileNotFoundError(f"No aligned JSONs in {aligned_dir}/<track>/")

    mano = _mano_layer(mano_assets_root)

    W, H = Image.open(frame_files[0]).size
    # Real intrinsics are constant across frames in an aligned set; pick from
    # any record. Far plane covers the actual cam_t.z range with margin.
    first = next(iter(records.values()))[0]
    fx = float(first["intrinsics"]["fx"])
    fy = float(first["intrinsics"]["fy"])
    cx = float(first["intrinsics"]["cx"])
    cy = float(first["intrinsics"]["cy"])
    z_max = max(rec["cam_t"][2] for recs in records.values() for rec in recs)
    zfar = max(20.0, float(z_max) * 1.5)
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
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
                verts_cv, faces = _build_mesh(rec, mano)
                verts_gl = verts_cv * _CV_TO_GL
                color = _TRACK_COLORS[(rec["track_id"] - 1) % len(_TRACK_COLORS)]
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
                color = _TRACK_COLORS[(rec["track_id"] - 1) % len(_TRACK_COLORS)]
                y = 8 + i * 24
                draw.rectangle([8, y, 28, y + 20], fill=color)
                draw.text((34, y),
                          f"id={rec['track_id']}  {'R' if rec['is_right'] else 'L'}",
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
    parser.add_argument("--aligned_dir",      required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--output_path",      required=True)
    parser.add_argument("--fps",   type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    args = parser.parse_args()
    render_hands_aligned_video(
        frames_dir       = args.frames_dir,
        aligned_dir      = args.aligned_dir,
        mano_assets_root = args.mano_assets_root,
        output_path      = args.output_path,
        fps              = args.fps,
        alpha            = args.alpha,
    )


if __name__ == "__main__":
    main()
