# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Refine SAM2 hand masks by intersecting with a dilated MANO silhouette.

For each (hand track, frame) where we have both a wilor pose record and a
SAM2 mask:

  1. Render the MANO mesh under wilor's virtual pinhole at that frame —
     binary silhouette ``S_render`` aligned to the source image pixels.
  2. Dilate ``S_render`` by ``dilation_pixels`` (forgiving when wilor's pose
     is slightly off the actual hand outline).
  3. Output mask = ``dilate(S_render) ∧ S_sam2``.

The intersection caps the SAM2 mask to "stuff wilor thinks is the hand,
plus a small buffer." Trims forearm bleed when SAM2's bbox prompt was
ambiguous near the wrist. Pixels SAM2 marked outside the dilated silhouette
are dropped; pixels SAM2 missed inside the silhouette are not added back
(the intent is to *constrain*, not expand).

Inputs:
  wilor_dir/<track_id>/<frame:06d>.json      per-track wilor records
                                              (from tracks_from_wilor_masks)
  masks_dir/<track_id>/<frame:06d>.png       SAM2 propagated masks
  tracks_path                                hand_tracks.json
  mano_assets_root                           manotorch ManoLayer asset root

Output:
  output_dir/<track_id>/<frame:06d>.png      refined binary masks

Usage:
    python -m v2d.wilor.lib.masks_intersect_silhouette \\
        --wilor_dir        /data/wilor \\
        --masks_dir        /data/masks \\
        --tracks_path      /data/hand_tracks.json \\
        --output_dir       /data/masks_refined \\
        --mano_assets_root /data/weights/wilor/pretrained_models \\
        --dilation_pixels  20
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Tuple

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import torch
import trimesh
from manotorch.manolayer import ManoLayer
from PIL import Image
from tqdm import tqdm


_CV_TO_GL = np.array([1.0, -1.0, -1.0])


def _mano_layer(mano_assets_root: str) -> ManoLayer:
    return ManoLayer(
        rot_mode         = "axisang",
        use_pca          = False,
        side             = "right",
        center_idx       = None,
        mano_assets_root = mano_assets_root,
    )


def _build_mesh(rec: dict, mano: ManoLayer) -> Tuple[np.ndarray, np.ndarray]:
    """MANO mesh in CV camera space + face indices."""
    mano_p = rec["mano"]
    pose_aa = np.concatenate([
        np.array(mano_p["global_orient"], dtype=np.float32),
        np.array(mano_p["hand_pose"],     dtype=np.float32),
    ])
    betas = np.array(mano_p["betas"], dtype=np.float32)
    out = mano(
        torch.from_numpy(pose_aa)[None],
        torch.from_numpy(betas)[None],
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


def _silhouette(
    rec: dict, mano: ManoLayer,
    renderer: pyrender.OffscreenRenderer, W: int, H: int,
) -> np.ndarray:
    """Render a single record's MANO mesh to a boolean silhouette (H, W)."""
    verts_cv, faces = _build_mesh(rec, mano)
    verts_gl = verts_cv * _CV_TO_GL
    focal = float(rec["camera"]["scaled_focal_length"])
    z_max = float(verts_cv[:, 2].max())
    zfar = max(50.0, z_max * 1.5)
    cam = pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=W / 2, cy=H / 2,
                                    znear=0.01, zfar=zfar)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1, 1, 1])
    scene.add(cam, pose=np.eye(4))
    tm = trimesh.Trimesh(verts_gl, faces, process=False)
    tm.visual.face_colors = np.array([255, 255, 255, 255], dtype=np.uint8)
    scene.add(pyrender.Mesh.from_trimesh(tm, smooth=False))
    rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    return rgba[:, :, 3] > 0


def masks_intersect_silhouette(
    wilor_dir: str,
    masks_dir: str,
    tracks_path: str,
    output_dir: str,
    mano_assets_root: str,
    dilation_pixels: int = 20,
) -> None:
    """Iterate hand tracks; for each (track, frame), write the refined mask
    ``dilate(rendered_silhouette, r) ∧ sam2_mask``.

    Frames missing either the wilor record or the SAM2 mask are skipped.
    The output directory tree mirrors ``masks_dir`` (one subdir per track).
    """
    with open(tracks_path) as f:
        tracks = json.load(f)["tracks"]
    hand_tracks = [t for t in tracks if t.get("role", "hand") == "hand"]
    if not hand_tracks:
        raise RuntimeError(f"No hand tracks in {tracks_path}")

    mano = _mano_layer(mano_assets_root)

    r = max(0, int(dilation_pixels))
    kernel = (np.ones((2 * r + 1, 2 * r + 1), dtype=np.uint8)
              if r > 0 else None)

    os.makedirs(output_dir, exist_ok=True)

    for track in hand_tracks:
        oid = int(track["object_id"])
        out_track = os.path.join(output_dir, str(oid))
        os.makedirs(out_track, exist_ok=True)

        wilor_track_dir = os.path.join(wilor_dir, str(oid))
        sam_track_dir   = os.path.join(masks_dir, str(oid))
        if not os.path.isdir(wilor_track_dir):
            print(f"  track {oid}: no wilor records at {wilor_track_dir}; skip")
            continue
        if not os.path.isdir(sam_track_dir):
            print(f"  track {oid}: no SAM2 masks at {sam_track_dir}; skip")
            continue

        # Index wilor records by frame for O(1) lookup; iterate over SAM2
        # mask frames so every SAM2 frame produces an output PNG. Frames
        # without a wilor record pass the SAM2 mask through unchanged so
        # downstream consumers (align, hamer, gsplat) always see *some*
        # mask wherever SAM2 said the hand was present.
        rec_by_frame: dict[int, str] = {}
        for p in sorted(glob.glob(os.path.join(wilor_track_dir, "*.json"))):
            try:
                f_idx = int(os.path.splitext(os.path.basename(p))[0])
            except ValueError:
                continue
            rec_by_frame[f_idx] = p
        if not rec_by_frame:
            print(f"  track {oid}: empty wilor dir; skip")
            continue

        sam_files = sorted(glob.glob(os.path.join(sam_track_dir, "*.png")))
        if not sam_files:
            print(f"  track {oid}: no SAM2 mask PNGs; skip")
            continue
        W, H = Image.open(sam_files[0]).size

        renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
        n_refined = n_passthrough = 0
        try:
            for sam_path in tqdm(sam_files, desc=f"  track {oid}",
                                 ncols=80, unit="frame"):
                stem = os.path.splitext(os.path.basename(sam_path))[0]
                try:
                    frame_idx = int(stem)
                except ValueError:
                    continue
                sam = np.asarray(Image.open(sam_path)) > 0
                rec_path = rec_by_frame.get(frame_idx)
                if rec_path is None:
                    # No wilor pose → pass SAM2 mask through unchanged.
                    refined = sam
                    n_passthrough += 1
                else:
                    with open(rec_path) as f:
                        rec = json.load(f)
                    sil = _silhouette(rec, mano, renderer, W, H).astype(np.uint8)
                    if kernel is not None:
                        sil = cv2.dilate(sil, kernel, iterations=1)
                    refined = (sil > 0) & sam
                    n_refined += 1
                out_path = os.path.join(out_track, f"{frame_idx:06d}.png")
                Image.fromarray((refined.astype(np.uint8) * 255), mode="L").save(out_path)
        finally:
            renderer.delete()

        print(f"  track {oid}: refined {n_refined}, "
              f"passed-through {n_passthrough} (no wilor record).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wilor_dir",        required=True)
    parser.add_argument("--masks_dir",        required=True)
    parser.add_argument("--tracks_path",      required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--dilation_pixels",  type=int, default=20,
                        help="Pixel radius of the square dilation kernel "
                             "applied to the rendered MANO silhouette before "
                             "intersecting with the SAM2 mask. 0 = no dilate "
                             "(exact silhouette intersection; aggressive).")
    args = parser.parse_args()
    masks_intersect_silhouette(
        wilor_dir        = args.wilor_dir,
        masks_dir        = args.masks_dir,
        tracks_path      = args.tracks_path,
        output_dir       = args.output_dir,
        mano_assets_root = args.mano_assets_root,
        dilation_pixels  = args.dilation_pixels,
    )


if __name__ == "__main__":
    main()
