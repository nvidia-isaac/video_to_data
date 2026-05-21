"""Render per-track per-frame MANO silhouette masks from HaMeR-style tracks.

Reads a directory of per-track per-frame JSONs and writes a binary mask PNG
per record. Two input schemas are auto-detected per record:

  * **Aligned** (output of ``v2d.hamer.align_hands`` or
    ``v2d.hamer.tracks_to_masks`` consumers): has top-level ``cam_t`` plus
    ``intrinsics`` and ``image_size``. Render at the record's own
    intrinsics — cam_t is already in real-camera units.

  * **Raw / pre-aligned** (output of ``v2d.wilor.*`` or
    ``v2d.hand_alignment.dynhamr_to_hamer_tracks``): has
    ``camera.pred_cam_t_full`` and ``camera.scaled_focal_length`` (the
    "virtual pinhole" — focal=scaled_focal, principal point at (W/2, H/2)).
    We rescale cam_t to real intrinsics using the same virtual-pinhole
    transform as ``align_hands`` (project under virtual, unproject under
    real at the rescaled depth), then render at real intrinsics.

Output layout:
    <output_dir>/<track_id>/<frame:06d>.png   # uint8, {0, 255}

Mask is the rendered MANO silhouette (``rendered_depth > 0``). No depth
intersection, no SAM2 mask gating — this is purely a geometry projection.

Usage
-----
    python -m v2d.hamer.lib.tracks_to_masks \\
        --tracks_dir       /data/dynhamr_as_hamer_tracks_moge \\
        --intrinsics_path  /data/intrinsics_stable.json \\
        --mano_assets_root /data/weights/hand \\
        --output_dir       /data/hand_silhouettes
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Tuple

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
from PIL import Image
from tqdm import tqdm

from v2d.hamer.lib.align_hands import (
    _CV_TO_GL_4X4,
    _load_intrinsics,
    _make_mano,
    _mesh_for_record,
)


def _render_silhouette(
    renderer: pyrender.OffscreenRenderer,
    cam: pyrender.IntrinsicsCamera,
    mesh,
    cam_t: np.ndarray,
) -> np.ndarray:
    """Return uint8 {0, 255} silhouette of ``mesh`` translated by ``cam_t``."""
    posed = mesh.copy()
    posed.vertices = posed.vertices + cam_t[None, :]
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(posed, smooth=False))
    scene.add(cam, pose=_CV_TO_GL_4X4)
    _, depth = renderer.render(scene)
    return ((depth > 0).astype(np.uint8) * 255)


def _resolve_cam_t(
    rec: dict,
    fx_real: float, fy_real: float, cx_real: float, cy_real: float,
    W: int, H: int,
) -> np.ndarray:
    """Return cam-frame translation in real-camera units.

    Aligned records (``cam_t`` present) pass through. Raw records get the
    same virtual-pinhole → real-pinhole rescaling as ``align_hands``.
    """
    if "cam_t" in rec:
        return np.asarray(rec["cam_t"], dtype=np.float64)
    scaled_focal = float(rec["camera"]["scaled_focal_length"])
    pred_cam_t_full = np.array(rec["camera"]["pred_cam_t_full"], dtype=np.float64)
    cx_v, cy_v = W / 2.0, H / 2.0
    z_v = float(pred_cam_t_full[2])
    if abs(z_v) < 1e-9:
        return pred_cam_t_full
    u_pix = scaled_focal * pred_cam_t_full[0] / z_v + cx_v
    v_pix = scaled_focal * pred_cam_t_full[1] / z_v + cy_v
    z_real = z_v * (fx_real / scaled_focal)
    x_real = (u_pix - cx_real) * z_real / fx_real
    y_real = (v_pix - cy_real) * z_real / fy_real
    return np.array([x_real, y_real, z_real], dtype=np.float64)


def tracks_to_masks(
    tracks_dir: str,
    intrinsics_path: str,
    mano_assets_root: str,
    output_dir: str,
) -> None:
    fx, fy, cx, cy, W, H = _load_intrinsics(intrinsics_path)
    print(f"Real intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}  {W}×{H}")

    mano = _make_mano(mano_assets_root)
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy,
                                    znear=0.01, zfar=100.0)
    renderer = pyrender.OffscreenRenderer(W, H)

    track_dirs = sorted(
        d for d in glob.glob(os.path.join(tracks_dir, "*")) if os.path.isdir(d)
    )
    if not track_dirs:
        raise FileNotFoundError(f"No track subdirs in {tracks_dir}")

    try:
        for track_dir in track_dirs:
            oid = os.path.basename(track_dir)
            files = sorted(glob.glob(os.path.join(track_dir, "*.json")))
            if not files:
                continue
            out_track = os.path.join(output_dir, oid)
            os.makedirs(out_track, exist_ok=True)
            print(f"\nTrack {oid}: {len(files)} frames → {out_track}")

            for src in tqdm(files, desc=f"  track {oid}", ncols=80, unit="frame"):
                frame_idx = int(os.path.splitext(os.path.basename(src))[0])
                out_path = os.path.join(out_track, f"{frame_idx:06d}.png")
                if os.path.exists(out_path):
                    continue
                with open(src) as f:
                    rec = json.load(f)
                mesh = _mesh_for_record(rec, mano)
                cam_t = _resolve_cam_t(rec, fx, fy, cx, cy, W, H)
                hand_scale = float(rec.get("hand_scale", 1.0))
                if hand_scale != 1.0:
                    centroid = mesh.vertices.mean(axis=0, keepdims=True)
                    mesh.vertices = (mesh.vertices - centroid) * hand_scale + centroid
                silhouette = _render_silhouette(renderer, cam, mesh, cam_t)
                Image.fromarray(silhouette).save(out_path)
    finally:
        renderer.delete()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracks_dir",       required=True,
                   help="HaMeR-style tracks dir (raw or aligned). Reads "
                        "<tracks_dir>/<track_id>/<frame:06d>.json.")
    p.add_argument("--intrinsics_path",  required=True,
                   help="Real-camera intrinsics JSON (fx, fy, cx, cy, width, height).")
    p.add_argument("--mano_assets_root", required=True)
    p.add_argument("--output_dir",       required=True,
                   help="Writes <output_dir>/<track_id>/<frame:06d>.png.")
    args = p.parse_args()
    tracks_to_masks(
        tracks_dir       = args.tracks_dir,
        intrinsics_path  = args.intrinsics_path,
        mano_assets_root = args.mano_assets_root,
        output_dir       = args.output_dir,
    )


if __name__ == "__main__":
    main()
