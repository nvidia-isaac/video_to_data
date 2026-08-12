# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assign WiLoR per-frame detections to SAM2 tracks via silhouette IoU.

WiLoR's detector runs independently per frame and produces an unordered list
of hand detections. To get a per-track per-frame layout (matching hamer's),
we render each detection's MANO mesh as a binary silhouette and IoU-match it
against the SAM2 propagated track masks.

Inputs:
  frames_dir/<frame:06d>.{png,jpg}             source frames (for image size)
  wilor_raw_dir/<frame:06d>.json               per-frame wilor detection lists
  masks_dir/<track_id>/<frame:06d>.png         SAM2 propagated masks
  tracks_path                                  hand_tracks.json (drives which
                                                track ids are "hand"
                                                candidates and the L/R label
                                                used at the seed frame)
  mano_assets_root                             dir containing
                                                models/MANO_RIGHT.pkl

Outputs:
  output_dir/<track_id>/<frame:06d>.json       per-track MANO record, hamer-
                                                compatible schema with
                                                ``track_id`` + ``frame_idx``
                                                added.

Assignment is greedy per frame: repeatedly pick the (detection, hand-track)
pair with highest IoU until below ``--min_iou``. Each detection is assigned
to at most one track and vice versa. Frames where wilor finds no detections,
or where no detection meets the IoU threshold, simply get no output (same
"hand not visible" convention as hamer's masks_to_hands).

Usage:
    python -m v2d.wilor.lib.tracks_from_wilor_masks \\
        --frames_dir       /data/frames \\
        --wilor_raw_dir    /data/wilor_raw \\
        --masks_dir        /data/masks \\
        --tracks_path      /data/hand_tracks.json \\
        --output_dir       /data/wilor \\
        --mano_assets_root /data/weights/wilor/pretrained_models
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
import torch
import trimesh
from manotorch.manolayer import ManoLayer
from PIL import Image
from tqdm import tqdm

_CV_TO_GL = np.array([1.0, -1.0, -1.0])


def _mano_layer(mano_assets_root: str) -> ManoLayer:
    return ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="right",
        center_idx=None,
        mano_assets_root=mano_assets_root,
    )


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


def _silhouette(
    rec: dict,
    mano: ManoLayer,
    renderer: pyrender.OffscreenRenderer,
    W: int,
    H: int,
) -> np.ndarray:
    """Render a single detection's MANO mesh to a boolean silhouette."""
    verts_cv, faces = _build_mesh_for_record(rec, mano)
    verts_gl = verts_cv * _CV_TO_GL
    focal = rec["camera"]["scaled_focal_length"]
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


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def _greedy_assign(iou_matrix: np.ndarray, min_iou: float) -> list[tuple[int, int]]:
    """Return list of (det_idx, track_idx) pairs by repeated max selection."""
    M = iou_matrix.copy()
    pairs: list[tuple[int, int]] = []
    while M.size > 0 and M.max() >= min_iou:
        k, t = np.unravel_index(M.argmax(), M.shape)
        pairs.append((int(k), int(t)))
        M[k, :] = -1.0
        M[:, t] = -1.0
    return pairs


def tracks_from_wilor_masks(
    frames_dir: str,
    wilor_raw_dir: str,
    masks_dir: str,
    tracks_path: str,
    output_dir: str,
    mano_assets_root: str,
    min_iou: float = 0.1,
) -> None:
    with open(tracks_path) as f:
        meta = json.load(f)
    hand_tracks = [t for t in meta["tracks"] if t.get("role", "hand") == "hand"]
    if not hand_tracks:
        raise RuntimeError(f"No hand tracks in {tracks_path}")
    hand_track_ids = [int(t["object_id"]) for t in hand_tracks]
    print(f"Hand track ids: {hand_track_ids}")

    frame_files = sorted(
        glob.glob(os.path.join(frames_dir, "*.png")) +
        glob.glob(os.path.join(frames_dir, "*.jpg"))
    )
    if not frame_files:
        raise FileNotFoundError(f"No frames in {frames_dir}")
    W, H = Image.open(frame_files[0]).size

    os.makedirs(output_dir, exist_ok=True)
    for tid in hand_track_ids:
        os.makedirs(os.path.join(output_dir, str(tid)), exist_ok=True)

    mano = _mano_layer(mano_assets_root)
    renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)

    n_written = 0
    n_unmatched_det = 0
    n_no_det = 0

    try:
        for frame_path in tqdm(frame_files, desc="match", ncols=80, unit="frame"):
            stem = os.path.splitext(os.path.basename(frame_path))[0]
            try:
                frame_idx = int(stem)
            except ValueError:
                continue
            wilor_json = os.path.join(wilor_raw_dir, f"{frame_idx:06d}.json")
            if not os.path.exists(wilor_json):
                n_no_det += 1
                continue
            with open(wilor_json) as f:
                dets = json.load(f)
            if not dets:
                n_no_det += 1
                continue

            # Load SAM2 masks for this frame; tracks with no mask this frame
            # are dropped from the candidate set.
            track_masks: list[tuple[int, np.ndarray]] = []
            for tid in hand_track_ids:
                p = os.path.join(masks_dir, str(tid), f"{frame_idx:06d}.png")
                if not os.path.exists(p):
                    continue
                m = np.asarray(Image.open(p)) > 0
                if m.sum() == 0:
                    continue
                track_masks.append((tid, m))
            if not track_masks:
                continue

            # Render each detection's silhouette.
            sils = [_silhouette(d, mano, renderer, W, H) for d in dets]

            iou = np.zeros((len(sils), len(track_masks)), dtype=np.float32)
            for k, s in enumerate(sils):
                for t, (_, m) in enumerate(track_masks):
                    iou[k, t] = _iou(s, m)

            pairs = _greedy_assign(iou, min_iou)
            matched_det_idx = {k for (k, _) in pairs}
            for det_idx, track_pos in pairs:
                tid = track_masks[track_pos][0]
                rec = dict(dets[det_idx])
                rec["track_id"]  = tid
                rec["frame_idx"] = frame_idx
                rec["match_iou"] = float(iou[det_idx, track_pos])
                with open(os.path.join(output_dir, str(tid), f"{frame_idx:06d}.json"), "w") as f:
                    json.dump(rec, f, indent=2)
                n_written += 1
            n_unmatched_det += (len(sils) - len(matched_det_idx))
    finally:
        renderer.delete()

    print(f"Wrote {n_written} per-track JSONs. Unmatched detections: "
          f"{n_unmatched_det}. Frames with no wilor detection: {n_no_det}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir",       required=True)
    parser.add_argument("--wilor_raw_dir",    required=True)
    parser.add_argument("--masks_dir",        required=True)
    parser.add_argument("--tracks_path",      required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--min_iou", type=float, default=0.1)
    args = parser.parse_args()
    tracks_from_wilor_masks(
        frames_dir       = args.frames_dir,
        wilor_raw_dir    = args.wilor_raw_dir,
        masks_dir        = args.masks_dir,
        tracks_path      = args.tracks_path,
        output_dir       = args.output_dir,
        mano_assets_root = args.mano_assets_root,
        min_iou          = args.min_iou,
    )


if __name__ == "__main__":
    main()
