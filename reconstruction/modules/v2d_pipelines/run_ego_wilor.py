# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WiLoR-driven hand pipeline: detect per frame → SAM2 → IoU match → per-track JSONs.

Differs from ``run_hand_masks.py`` in two ways:
  * the hand detector is WiLoR (per-frame, end-to-end bbox + MANO) rather
    than MediaPipe on the reference frame only;
  * per-track per-frame MANO assignment is done by rendering each WiLoR
    detection's MANO mesh and IoU-matching it against the propagated SAM2
    masks, rather than running a separate pose model on mask-cropped patches.

Steps:
  0.  AnyCalib undistortion           [optional, --undistort]
  0b. GeoCalib intrinsics + gravity    [optional, --run_geocalib/--gravity_align]
  1.  Extract frames
  2.  MoGe depth + intrinsics         → depth/ + intrinsics_stable.json
  2c. DROID-SLAM trajectory           [optional, --run_slam]
                                       → slam_poses/<frame:06d>.json
                                       (scale-aligned to MoGe depth; seeds
                                       gsplat's background pose field
                                       at step 15 when --run_refinement)
  3.  WiLoR over all frames           → wilor_raw/<frame:06d>.json
  4.  (Object) Grounding DINO          [only when --object_prompt is set]
  5.  Build SAM2 prompts (object + WiLoR hand box/mask prompts)
                                       → sam2_prompts.json + hand_tracks.json
  6.  SAM2 mask propagation            → masks/{1,2,…}/*.png
  7.  Render prompts overlay + masks overlay
  8.  Object branch (SAM3D + FoundationPose + EKF)   [--object_prompt]
  9.  Match wilor detections to SAM2 tracks via silhouette IoU
                                       → wilor/<track_id>/<frame:06d>.json
  9b. (Optional) Fill wilor track gaps pre-align (every visible frame)
                                       → wilor_filled/<track_id>/<frame:06d>.json
                                       [--refine_masks_with_silhouette]
  9c. (Optional) Refine SAM2 hand masks via dilated MANO silhouette ∩
                                       → masks_refined/<track_id>/*.png
                                       [--refine_masks_with_silhouette]
  10. Render virtual-cam overlay (raw) → wilor_overlay.mp4
  11. Depth-align hands, real frames only (real intrinsics)
                                       → wilor_aligned/<track_id>/<frame:06d>.json
  12. Render aligned overlay (+ object if --object_prompt)
                                       → wilor_aligned_overlay.mp4
  13. Interpolate missing aligned frames
                                       → wilor_aligned_filled/<track_id>/<frame:06d>.json
                                         (tagged `interpolated: true|false`)
  14. Render filled aligned overlay     → wilor_aligned_filled_overlay.mp4
  14a-e. (Optional) HaMeR pass         [--run_hamer_pass]
                                       Per-track HaMeR regression on the
                                       pass-1 SAM2 masks + alignment +
                                       interpolation:
                                       → hamer/, hamer_aligned/,
                                         hamer_aligned_filled/, *_overlay.mp4
  15. Joint hand+object GS refinement   [--run_refinement, requires --object_prompt]
                                       → poses_refined/, wilor_refined/, *_overlay.mp4

Output directory layout:
  <output_dir>/
  ├── frames/
  ├── depth/                           # MoGe depth PNGs
  ├── intrinsics/                      # MoGe per-frame intrinsics JSONs
  ├── intrinsics_stable.json
  ├── geocalib/                       # [--run_geocalib or --gravity_align]
  ├── slam_poses/<frame:06d>.json      # [--run_slam] DROID-SLAM cam-to-world
  ├── slam_trajectory.txt              # [--run_slam] TUM format
  ├── wilor_raw/<frame:06d>.json
  ├── hand_detections.json             # ref-frame slice of wilor_raw
  ├── sam2_prompts.json
  ├── sam2_hand_prompt_masks/          # [--sam2_hand_prompt_source wilor_mask]
  ├── hand_tracks.json
  ├── masks/{...}/*.png
  ├── prompts_overlay.png
  ├── masks_overlay.mp4
  ├── wilor/{...}/<frame:06d>.json                  # real detections only
  ├── wilor_overlay.mp4
  ├── wilor_aligned/{...}/<frame:06d>.json          # aligned real frames
  ├── wilor_aligned_overlay.mp4
  ├── wilor_aligned_filled/{...}/<frame:06d>.json   # real + interpolated (tagged)
  ├── wilor_aligned_filled_overlay.mp4
  └── result/                                      # Portable final bundle when object branch runs
                                                   # [--gravity_align] world transforms rotated to target gravity

Usage:
    python modules/v2d_pipelines/run_ego_wilor.py \\
        --video_path data/clip.mp4 \\
        --output_dir data/outputs/clip \\
        --sam2_weights  data/weights/sam2 \\
        --wilor_weights data/weights/wilor \\
        --undistort

    # Use an existing metric OBJ instead of SAM3D mesh generation + scale search:
    python modules/v2d_pipelines/run_ego_wilor_with_obj.py \\
        --video_path data/clip.mp4 \\
        --output_dir data/outputs/clip_obj \\
        --object_prompt "red mug" \\
        --object_mesh_path data/assets/red_mug.obj

Run from reconstruction/.
"""

import argparse
import glob
import json
import os
import subprocess
import tempfile

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


def _apply_sam3d_transform(mesh_path: str, transform_path: str, out_path: str) -> None:
    """Apply SAM3D rotation+scale (no translation) to mesh vertices."""
    with open(transform_path) as f:
        t = json.load(f)
    qw, qx, qy, qz = t["rotation"]
    sx, sy, sz      = t["scale"]
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz,  2*qx*qy - 2*qw*qz,      2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz,      1 - 2*qx*qx - 2*qz*qz,  2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy,      2*qy*qz + 2*qw*qx,      1 - 2*qx*qx - 2*qy*qy],
    ], dtype=np.float64)
    RS = R @ np.diag([sx, sy, sz])
    scene = trimesh.load(mesh_path)
    if isinstance(scene, trimesh.Scene):
        meshes = list(scene.geometry.values())
        for m in meshes:
            m.vertices = (RS @ m.vertices.T).T
        result = trimesh.util.concatenate(meshes)
    else:
        scene.vertices = (RS @ scene.vertices.T).T
        result = scene
    result.export(out_path)


from v2d.anycalib.docker.run_video_to_calibration import (
    run_video_to_calibration as run_anycalib_video_to_calibration,
)
from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.common.result_bundle import (
    gravity_align_result_bundle,
    result_bundle_has_gravity_alignment,
    write_result_bundle,
)
from v2d.common.utils import extract_images
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.droid_slam.docker.run_video_to_slam import run_video_to_slam
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.geocalib.docker.run_video_to_calibration import (
    run_video_to_calibration as run_geocalib_video_to_calibration,
)
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.gsplat_refinement.docker.run_refine import run_refine
from v2d.gsplat_refinement.docker.run_refine_simple import run_refine_simple
from v2d.hamer.docker.run_align_hands import run_align_hands
from v2d.hamer.docker.run_masks_to_hands import run_masks_to_hands as run_hamer_masks_to_hands
from v2d.hamer.docker.run_render_hands_aligned_video import run_render_hands_aligned_video
from v2d.hamer.docker.run_render_hands_video import run_render_hands_video as run_hamer_render
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_moge_depth
from v2d.pipelines.run_hand_masks import _plot_pose_grad_from_checkpoint
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
from v2d.wilor.docker.run_masks_intersect_silhouette import run_masks_intersect_silhouette
from v2d.wilor.docker.run_render_hands_video import run_render_hands_video as run_wilor_render
from v2d.wilor.docker.run_render_wilor_prompt_masks import run_render_wilor_prompt_masks
from v2d.wilor.docker.run_tracks_from_wilor_masks import run_tracks_from_wilor_masks
from v2d.wilor.docker.run_tracks_interpolate import run_tracks_interpolate
from v2d.wilor.docker.run_video_to_hands import run_video_to_hands as run_wilor_video
from v2d.wilor.docker.run_wilor_tracks_interpolate import run_wilor_tracks_interpolate


_TRACK_COLORS = [
    (255,  60,  60), ( 60, 160, 255), ( 60, 200,  60),
    (255, 180,  40), (200,  80, 220), ( 60, 220, 220),
    (255, 240,  60), (240, 100, 180), (180, 220, 100), (140, 110, 80),
]


def _track_color(object_id: int) -> tuple[int, int, int]:
    return _TRACK_COLORS[(object_id - 1) % len(_TRACK_COLORS)]


def _font(size: int = 18):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _render_prompts_overlay(
    image_path: str, tracks: list[dict], detections: list[dict], output_path: str,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(20)
    for track, det in zip(tracks, detections):
        oid   = track["object_id"]
        color = _track_color(oid)
        bb    = det["bbox"]
        x0, y0, x1, y1 = bb["x0"], bb["y0"], bb["x1"], bb["y1"]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        label = f"id={oid}  {'R' if track['is_right'] else 'L'}  {track['score']:.2f}"
        tw = draw.textlength(label, font=font)
        draw.rectangle([x0, y0 - 26, x0 + tw + 8, y0], fill=color)
        draw.text((x0 + 4, y0 - 24), label, fill=(0, 0, 0), font=font)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    img.save(output_path)
    print(f"  Wrote prompts overlay   → {output_path}")


def _render_masks_overlay(
    frames_dir: str,
    masks_dir: str,
    tracks: list[dict],
    output_path: str,
    fps: float = 30.0,
    alpha: float = 0.45,
) -> None:
    frame_files = sorted(
        glob.glob(os.path.join(frames_dir, "*.png")) +
        glob.glob(os.path.join(frames_dir, "*.jpg"))
    )
    if not frame_files:
        raise FileNotFoundError(f"No frames in {frames_dir}")

    track_ids = [t["object_id"] for t in tracks]
    label_font = _font(18)

    with tempfile.TemporaryDirectory() as tmpdir:
        for f_idx, frame_path in enumerate(frame_files):
            frame = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32)
            for oid in track_ids:
                mask_path = os.path.join(masks_dir, str(oid), f"{f_idx:06d}.png")
                if not os.path.exists(mask_path):
                    continue
                mask = np.asarray(Image.open(mask_path), dtype=np.float32)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                if mask.max() <= 1.0:
                    mask = mask * 255.0
                m = (mask > 127).astype(np.float32)[..., None]
                color = np.array(_track_color(oid), dtype=np.float32)[None, None, :]
                frame = frame * (1 - alpha * m) + color * (alpha * m)
            out_img = Image.fromarray(frame.clip(0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(out_img)
            for i, t in enumerate(tracks):
                y = 8 + i * 24
                color = _track_color(t["object_id"])
                draw.rectangle([8, y, 28, y + 20], fill=color)
                draw.text((34, y), f"id={t['object_id']}  {'R' if t['is_right'] else 'L'}",
                          fill=(255, 255, 255), font=label_font,
                          stroke_width=2, stroke_fill=(0, 0, 0))
            out_img.save(os.path.join(tmpdir, f"{f_idx:06d}.png"))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-r", str(fps),
            "-i", os.path.join(tmpdir, "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            output_path,
        ], check=True)
    print(f"  Wrote masks overlay     → {output_path}")


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


def _step(label: str, done: bool) -> bool:
    if done:
        print(f"  [skip] {label}")
        return True
    print(f"  [run ] {label}")
    return False


def _reconcile_hand_tracks_from_handedness(
    hand_tracks_path: str,
    handedness_path: str,
) -> None:
    """Update hand_tracks.json from a tracks_interpolate handedness summary."""
    if not os.path.exists(handedness_path):
        return
    with open(handedness_path) as f:
        canonical = json.load(f)
    with open(hand_tracks_path) as f:
        ht = json.load(f)
    changed = False
    for t in ht.get("tracks", []):
        tid_key = str(t["object_id"])
        entry = canonical.get(tid_key)
        if entry is None:
            continue
        new_ir = bool(entry["is_right"])
        if bool(t["is_right"]) != new_ir:
            print(f"  hand_tracks.json: track {tid_key} is_right "
                  f"{t['is_right']} -> {new_ir} (sequence vote: "
                  f"R={entry['votes_right']} L={entry['votes_left']}, "
                  f"tiebroke={entry.get('tiebroke', False)})")
            t["is_right"] = new_ir
            changed = True
    if changed:
        with open(hand_tracks_path, "w") as f:
            json.dump(ht, f, indent=2)
        print(f"  Updated {hand_tracks_path} with canonical handedness")


def _resolve_prompt_mask_path(mask_path: str, prompts_path: str) -> str:
    if os.path.isabs(mask_path):
        return mask_path
    return os.path.join(os.path.dirname(os.path.abspath(prompts_path)), mask_path)


def _sam2_prompt_artifacts_match(
    sam2_prompts_path: str,
    hand_tracks_path: str,
    expected_hand_prompt_source: str,
) -> bool:
    if not (os.path.exists(sam2_prompts_path) and os.path.exists(hand_tracks_path)):
        return False
    try:
        with open(hand_tracks_path) as f:
            track_meta = json.load(f)
        with open(sam2_prompts_path) as f:
            prompts_meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    existing_source = track_meta.get("sam2_hand_prompt_source", "box")
    if existing_source != expected_hand_prompt_source:
        return False

    hand_ids = {
        int(t["object_id"])
        for t in track_meta.get("tracks", [])
        if t.get("role", "hand") == "hand"
    }
    prompts_by_id = {
        int(p["object_id"]): p
        for p in prompts_meta.get("prompts", [])
        if "object_id" in p
    }
    if not hand_ids:
        return False

    if expected_hand_prompt_source == "wilor_mask":
        for oid in hand_ids:
            mask_path = prompts_by_id.get(oid, {}).get("mask_path")
            if not mask_path:
                return False
            if not os.path.exists(_resolve_prompt_mask_path(mask_path, sam2_prompts_path)):
                return False
    else:
        for oid in hand_ids:
            prompt = prompts_by_id.get(oid, {})
            if prompt.get("mask_path") or not prompt.get("box"):
                return False
    return True


def _load_scale_json(path: str, default: float = 1.0) -> float:
    if not os.path.exists(path):
        return float(default)
    with open(path) as f:
        return float(json.load(f).get("scale", default))


def _first_populated_dir(directories: list[str | None]) -> str | None:
    for directory in directories:
        if directory and _has_files(directory):
            return directory
    return None


def _result_hand_dirs(
    hand_root: str | None,
    hand_tracks_path: str,
) -> tuple[str | None, str | None]:
    if not hand_root or not os.path.isdir(hand_root):
        return None, None
    try:
        with open(hand_tracks_path) as f:
            tracks = json.load(f).get("tracks", [])
    except (OSError, json.JSONDecodeError):
        return None, None

    left_ids = [
        int(t["object_id"])
        for t in tracks
        if t.get("role", "hand") == "hand" and not bool(t.get("is_right"))
    ]
    right_ids = [
        int(t["object_id"])
        for t in tracks
        if t.get("role", "hand") == "hand" and bool(t.get("is_right"))
    ]

    def _first_existing(track_ids: list[int]) -> str | None:
        for track_id in track_ids:
            candidate = os.path.join(hand_root, str(track_id))
            if _has_files(candidate):
                return candidate
        return None

    return _first_existing(left_ids), _first_existing(right_ids)


def _effective_init_opacity(
    specific: float | None,
    shared: float | None,
    default: float,
) -> float:
    value = specific if specific is not None else shared
    return float(default if value is None else value)


def _pad_bbox(
    bbox: dict, pad_ratio: float, img_w: int, img_h: int, square: bool = True,
) -> dict:
    """Center-pad a bbox, optionally squaring at the longer side."""
    x0, y0, x1, y1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    width = x1 - x0
    height = y1 - y0
    if square:
        width = height = max(width, height)
    half_w = width * (1.0 + pad_ratio) / 2.0
    half_h = height * (1.0 + pad_ratio) / 2.0
    return {
        "x0": max(0.0,          cx - half_w),
        "y0": max(0.0,          cy - half_h),
        "x1": min(float(img_w), cx + half_w),
        "y1": min(float(img_h), cy + half_h),
    }


def run_ego_wilor(
    video_path: str,
    output_dir: str,
    sam2_weights: str,
    wilor_weights: str = "data/weights/wilor",
    reference_frame: int = 0,
    sam2_bbox_pad: float = 0.0,
    sam2_square_wilor_bboxes: bool = True,
    sam2_hand_prompt_source: str = "box",
    undistort: bool = False,
    anycalib_weights: str = "data/weights/anycalib",
    run_geocalib: bool = False,
    geocalib_weights_path: str = "data/weights/geocalib",
    geocalib_weights: str = "pinhole",
    geocalib_camera_model: str = "pinhole",
    geocalib_num_samples: int = 8,
    use_geocalib_intrinsics: bool = False,
    gravity_align: bool = False,
    gravity_align_target: str = "z_up",
    moge_weights: str = "data/weights/moge",
    min_iou: float = 0.1,
    mask_min_pixels: int = 256,
    interp_betas: str = "fixed",
    interp_max_gap_frames: int = 15,
    object_prompt: str | None = None,
    object_mesh_path: str | None = None,
    grounding_dino_weights: str = "data/weights/grounding_dino",
    sam3d_weights: str = "data/weights/sam3d",
    foundation_pose_weights: str = "data/weights/foundation_pose",
    reregister_iou_thresh: float | None = 0.3,
    run_slam: bool = False,
    droid_slam_weights: str = "data/weights/droid_slam",
    bg_init_stride: int = 10,
    bg_voxel_size: float = 0.005,
    bg_max_points: int = 40000,
    w_scale_aniso_bg: float = 0.1,
    w_density_bg: float = 0.0,
    n_density_neighbors: int = 8,
    density_subsample_frac_bg: float = 0.2,
    # Paper-faithful SuGaR (Guédon & Lepetit 2024) — bg only. Anchors
    # Gaussians to MoGe depth via implicit SDF. Defaults chosen as a
    # reasonable starting point; tune up if bg still drifts off-surface.
    w_sdf_density_bg: float = 0.5,
    w_normal_consistency_bg: float = 0.1,
    n_sdf_samples_bg: int = 20000,
    n_sdf_neighbors_bg: int = 16,
    # SSIM photometric + depth (1 - SSIM on log-depth) — capture local
    # structure that pixel-wise L1 misses. 3DGS pairs photometric L1 with
    # SSIM at ratio ~0.8 / 0.2; we default to 0.2 photometric SSIM and
    # leave depth SSIM off by default (depth supervision itself is
    # currently disabled in pass 1).
    w_photometric_ssim: float = 0.2,
    w_depth_ssim: float = 0.0,
    # Static valid-pixel mask: drops fisheye black corners / dead-border
    # pixels from all supervision (photometric, depth, silhouette via the
    # union mask, SuGaR bg). Derived from per-pixel max brightness across
    # the input video — non-degenerate because there's one mask shared
    # across all frames and it's read directly from the input, not learned.
    valid_mask_threshold: float = 0.04,
    valid_mask_erode_iters: int = 2,
    refine_masks_with_silhouette: bool = False,
    mask_dilation_pixels: int = 20,
    hand_pose_source: str = "wilor",
    run_hamer_pass: bool = False,
    hamer_weights: str = "data/weights/hamer",
    hamer_bbox_expansion: float = 1.7,
    hamer_mask_min_pixels: int | None = None,
    run_refinement: bool = False,
    run_refinement_simple: bool = False,
    simple_refinement_epochs: int = 20,
    simple_refinement_batch_size: int = 16,
    simple_refinement_train_resolution_scale: float = 0.5,
    simple_refinement_lr_gaussians: float = 1e-3,
    simple_refinement_lr_object_pose: float = 1e-3,
    simple_refinement_lr_object_scale: float = 0.0,
    simple_refinement_lr_hand_pose: float = 1e-3,
    simple_refinement_lr_hand_articulation: float = 0.0,
    simple_refinement_lr_hand_shape: float = 0.0,
    simple_refinement_lr_hand_scale: float = 0.0,
    simple_refinement_lr_camera_pose: float = 1e-3,
    simple_refinement_lr_schedule: str = "constant",
    simple_refinement_lr_cosine_min_factor: float = 0.0,
    simple_refinement_w_smooth_object_rot: float = 0.1,
    simple_refinement_w_smooth_object_trans: float = 0.1,
    simple_refinement_w_smooth_hand_rot: float = 0.1,
    simple_refinement_w_smooth_hand_articulation: float = 0.1,
    simple_refinement_w_smooth_hand_trans: float = 0.1,
    simple_refinement_w_smooth_hand_object_relative_rot: float = 0.0,
    simple_refinement_w_smooth_hand_object_relative_trans: float = 0.0,
    simple_refinement_w_smooth_camera_rot: float = 0.1,
    simple_refinement_w_smooth_camera_trans: float = 0.1,
    simple_refinement_w_mask: float = 1.0,
    simple_refinement_w_relative_depth: float = 0.0,
    simple_refinement_w_perceptual: float = 0.0,
    simple_refinement_vgg_weights_path: str | None = None,
    simple_refinement_perceptual_resize: int = 224,
    simple_refinement_w_hand_object_penetration: float = 0.0,
    simple_refinement_hand_object_penetration_margin: float = 0.003,
    simple_refinement_object_sdf_resolution: int = 96,
    simple_refinement_hand_object_penetration_max_verts: int = 0,
    simple_refinement_mask_background: bool = False,
    simple_refinement_render_every: int = 25,
    simple_refinement_debug_frame: int | None = None,
    learn_hand_scale: bool = True,
    lr_hand_scale: float = 1e-3,
    w_hand_scale_prior: float = 10.0,
    refinement_render_every: int = 25,
    w_smooth_obj_rot: float = 1.0,
    w_smooth_obj_trans: float = 1.0,
    w_smooth_hand_rot: float = 1.0,
    w_smooth_hand_finger: float = 1.0,
    w_smooth_hand_trans: float = 1.0,
    w_smooth_bg_rot: float = 1.0,
    w_smooth_bg_trans: float = 1.0,
    smooth_obj_in_world: bool = False,
    smooth_hand_in_world: bool = False,
    rotation_search_smoothness_weight: float = 1.0,
    init_opacity: float | None = None,
    init_opacity_obj: float | None = None,
    init_opacity_hand: float | None = None,
    init_opacity_bg: float | None = None,
    init_opacity_wrist: float | None = None,
    init_gaussian_scale_factor: float = 1.0,
    refinement_resume: bool = False,
    refinement_object_pose_only: bool = False,
    refinement_random_init_obj_pose: bool = False,
    refinement_ignore_optimizer_state: bool = False,
    refinement_second_pass: bool = False,
    refinement_second_pass_lr_scale: float = 0.3,
    refinement_second_pass_epochs: int | None = None,
    refinement_second_pass_batch_size: int | None = None,
    dev: bool = False,
) -> None:
    video_path = os.path.abspath(video_path)
    output_dir = os.path.abspath(output_dir)
    if object_mesh_path is not None:
        object_mesh_path = os.path.abspath(object_mesh_path)
        if object_prompt is None:
            raise ValueError("--object_mesh_path requires --object_prompt for object masks.")
        if not os.path.isfile(object_mesh_path):
            raise FileNotFoundError(f"Object mesh OBJ not found: {object_mesh_path}")
    if simple_refinement_vgg_weights_path is not None:
        simple_refinement_vgg_weights_path = os.path.abspath(simple_refinement_vgg_weights_path)
        os.makedirs(
            os.path.dirname(simple_refinement_vgg_weights_path) or ".",
            exist_ok=True,
        )
    if sam2_hand_prompt_source not in {"box", "wilor_mask"}:
        raise ValueError(
            "sam2_hand_prompt_source must be 'box' or 'wilor_mask', "
            f"got {sam2_hand_prompt_source!r}"
        )
    if hand_pose_source not in {"wilor", "hamer"}:
        raise ValueError(
            f"hand_pose_source must be 'wilor' or 'hamer', got {hand_pose_source!r}"
        )
    if hand_pose_source == "hamer" and refine_masks_with_silhouette:
        raise ValueError(
            "--refine_masks_with_silhouette requires --hand_pose_source wilor; "
            "the HaMeR source mode uses SAM2 masks directly."
        )
    if geocalib_camera_model != "pinhole" and geocalib_weights != "distorted":
        print("  WARNING: GeoCalib distortion camera models are intended to use "
              "--geocalib_weights distorted.")
    run_geocalib_eff = run_geocalib or use_geocalib_intrinsics or gravity_align
    os.makedirs(output_dir, exist_ok=True)
    using_provided_object_mesh = object_mesh_path is not None
    use_hamer_primary = hand_pose_source == "hamer"
    run_hamer_outputs = run_hamer_pass or use_hamer_primary
    if hamer_mask_min_pixels is None:
        # Primary HaMeR mode is intended to produce a pose for every non-empty
        # SAM2 hand mask. The optional sidecar pass keeps the older noise guard.
        hamer_mask_min_pixels = 1 if use_hamer_primary else 256
    init_opacity_obj_eff = _effective_init_opacity(init_opacity_obj, init_opacity, 0.9)
    init_opacity_hand_eff = _effective_init_opacity(init_opacity_hand, init_opacity, 0.9)
    init_opacity_bg_eff = _effective_init_opacity(init_opacity_bg, init_opacity, 0.9)
    init_opacity_wrist_eff = _effective_init_opacity(init_opacity_wrist, init_opacity, 0.5)

    frames_dir         = f"{output_dir}/frames"
    depth_dir          = f"{output_dir}/depth"
    intrinsics_dir     = f"{output_dir}/intrinsics"
    intrinsics_stable  = f"{output_dir}/intrinsics_stable.json"
    anycalib_dir       = f"{output_dir}/anycalib"
    anycalib_intr      = f"{anycalib_dir}/intrinsics.json"
    anycalib_dist      = f"{anycalib_dir}/distortion.json"
    undistorted_video  = f"{anycalib_dir}/undistorted.mp4"
    undistorted_intr   = f"{anycalib_dir}/undistorted_intrinsics.json"
    geocalib_dir       = f"{output_dir}/geocalib"
    geocalib_intr      = f"{geocalib_dir}/intrinsics.json"
    geocalib_dist      = f"{geocalib_dir}/distortion.json"
    geocalib_gravity   = f"{geocalib_dir}/gravity.json"
    geocalib_calib     = f"{geocalib_dir}/calibration.json"
    wilor_raw_dir      = f"{output_dir}/wilor_raw"
    hand_detections    = f"{output_dir}/hand_detections.json"
    dino_detections    = f"{output_dir}/dino_detections.json"
    sam2_prompts       = f"{output_dir}/sam2_prompts.json"
    sam2_hand_prompt_masks_dir = f"{output_dir}/sam2_hand_prompt_masks"
    hand_tracks        = f"{output_dir}/hand_tracks.json"
    object_track       = f"{output_dir}/object_track.json"
    masks_dir          = f"{output_dir}/masks"
    masks_refined_dir  = f"{output_dir}/masks_refined"
    masks_refined_overlay = f"{output_dir}/masks_refined_overlay.mp4"
    prompts_overlay    = f"{output_dir}/prompts_overlay.png"
    masks_overlay      = f"{output_dir}/masks_overlay.mp4"
    # Object-branch outputs (only used when --object_prompt is set)
    mesh_dir           = f"{output_dir}/mesh"
    mesh_path          = f"{mesh_dir}/textured_mesh.obj"
    mesh_transform     = f"{mesh_dir}/mesh_transform.json"
    mesh_intrinsics    = f"{mesh_dir}/mesh_intrinsics.json"
    mesh_pretransformed= f"{output_dir}/mesh_pretransformed.obj"
    mesh_scaled        = f"{output_dir}/mesh_scaled.obj"
    scale_path         = f"{output_dir}/scale.json"
    object_mesh_for_pipeline: str | None = None
    poses_dir          = f"{output_dir}/poses"
    poses_smooth_dir   = f"{output_dir}/poses_smoothed"
    slam_poses_dir            = f"{output_dir}/slam_poses"
    slam_trajectory           = f"{output_dir}/slam_trajectory.txt"
    # Hand outputs
    wilor_tracks_dir          = f"{output_dir}/wilor"
    wilor_filled_dir          = f"{output_dir}/wilor_filled"
    wilor_overlay             = f"{output_dir}/wilor_overlay.mp4"
    wilor_aligned_dir         = f"{output_dir}/wilor_aligned"
    wilor_aligned_overlay     = f"{output_dir}/wilor_aligned_overlay.mp4"
    wilor_aligned_filled_dir  = f"{output_dir}/wilor_aligned_filled"
    wilor_aligned_filled_overlay = f"{output_dir}/wilor_aligned_filled_overlay.mp4"
    # HaMeR pass (runs on the pass-1 SAM2 masks).
    hamer_dir                 = f"{output_dir}/hamer"
    hamer_overlay             = f"{output_dir}/hamer_overlay.mp4"
    hamer_aligned_dir         = f"{output_dir}/hamer_aligned"
    hamer_aligned_overlay     = f"{output_dir}/hamer_aligned_overlay.mp4"
    hamer_aligned_filled_dir  = f"{output_dir}/hamer_aligned_filled"
    hamer_aligned_filled_overlay = f"{output_dir}/hamer_aligned_filled_overlay.mp4"
    ref_rgb            = f"{frames_dir}/{reference_frame:06d}.png"
    ref_depth          = f"{depth_dir}/{reference_frame:06d}.png"
    mano_assets_root   = os.path.join(wilor_weights, "pretrained_models")

    print(f"\n{'='*60}")
    print(f"  video           : {os.path.basename(video_path)}")
    print(f"  output          : {output_dir}")
    print(f"  reference_frame : {reference_frame}")
    print(f"  undistort       : {undistort}")
    print(f"  run_geocalib    : {run_geocalib_eff}")
    print(f"  gravity_align   : {gravity_align} ({gravity_align_target})")
    print(f"  object_prompt   : {object_prompt}")
    print(f"  sam2_hand_prompt_source: {sam2_hand_prompt_source}")
    print(f"  hand_pose_source: {hand_pose_source}")
    if object_mesh_path is not None:
        print(f"  object_mesh     : {object_mesh_path}")
    print(f"  run_slam        : {run_slam}")
    print(f"  min_iou         : {min_iou}")
    print(f"{'='*60}\n")

    # 0. AnyCalib undistortion (optional) ------------------------------------
    if undistort:
        os.makedirs(anycalib_dir, exist_ok=True)
        if not _step("AnyCalib undistortion", os.path.exists(undistorted_video)):
            run_anycalib_video_to_calibration(
                video_path                  = video_path,
                intrinsics_path             = anycalib_intr,
                distortion_path             = anycalib_dist,
                weights_path                = anycalib_weights,
                undistorted_video_path      = undistorted_video,
                undistorted_intrinsics_path = undistorted_intr,
                dev                         = dev,
            )
        video_path = undistorted_video
        print(f"  → using undistorted video: {video_path}")

    # 0b. GeoCalib intrinsics + gravity (optional) -------------------------
    if run_geocalib_eff:
        os.makedirs(geocalib_dir, exist_ok=True)
        geocalib_done = (
            os.path.exists(geocalib_intr)
            and os.path.exists(geocalib_dist)
            and os.path.exists(geocalib_gravity)
            and os.path.exists(geocalib_calib)
        )
        if not _step("GeoCalib intrinsics + gravity", geocalib_done):
            run_geocalib_video_to_calibration(
                video_path       = video_path,
                intrinsics_path  = geocalib_intr,
                distortion_path  = geocalib_dist,
                gravity_path     = geocalib_gravity,
                calibration_path = geocalib_calib,
                weights_path     = geocalib_weights_path,
                weights          = geocalib_weights,
                camera_model     = geocalib_camera_model,
                num_samples      = geocalib_num_samples,
                dev              = dev,
            )

    # 1. Extract frames ------------------------------------------------------
    if not _step("Extract frames", _has_files(frames_dir)):
        extract_images(video_path, frames_dir)

    # 2. MoGe depth + intrinsics --------------------------------------------
    # Needed for both the depth-alignment step (hands) and, when set, the
    # object branch (SAM3D + FoundationPose).
    moge_input_intrinsics = (
        geocalib_intr
        if use_geocalib_intrinsics
        else (undistorted_intr if undistort else None)
    )
    if not _step("MoGe depth + intrinsics", _has_files(depth_dir)):
        run_moge_depth(
            video_path            = video_path,
            depth_folder          = depth_dir,
            intrinsics_folder     = intrinsics_dir,
            weights_path          = moge_weights,
            input_intrinsics_path = moge_input_intrinsics,
            dev                   = dev,
        )
    if not _step("Stabilise intrinsics", os.path.exists(intrinsics_stable)):
        stabilize_intrinsics(intrinsics_dir, intrinsics_stable)

    # 2c. Optional DROID-SLAM ----------------------------------------------
    # Per-frame camera-to-world Transform3d JSONs. Scale-aligned to MoGe
    # depth so the trajectory is in metric units consistent with the rest of
    # the pipeline. Used downstream to seed gsplat's background pose field
    # (see step 15) — the identity-init basin in gsplat is only good for
    # near-static cameras; ego sequences need this seed.
    if run_slam:
        if not _step("DROID-SLAM trajectory", _has_files(slam_poses_dir)):
            run_video_to_slam(
                video_path            = video_path,
                poses_folder          = slam_poses_dir,
                weights_path          = droid_slam_weights,
                input_intrinsics_path = intrinsics_stable,
                align_to_depth_folder = depth_dir,
                trajectory_path       = slam_trajectory,
                dev                   = dev,
            )

    # 3. WiLoR over all frames -----------------------------------------------
    if not _step("WiLoR per-frame detection + MANO",
                 _has_files(wilor_raw_dir) and
                 os.path.exists(os.path.join(wilor_raw_dir, f"{reference_frame:06d}.json"))):
        run_wilor_video(
            video_path  = video_path,
            output_dir  = wilor_raw_dir,
            weights_dir = wilor_weights,
            dev         = dev,
        )

    # Slice out the ref-frame detections for SAM2 seeding + overlay legend.
    ref_wilor = os.path.join(wilor_raw_dir, f"{reference_frame:06d}.json")
    if not os.path.exists(ref_wilor):
        raise RuntimeError(
            f"WiLoR produced no detection file for reference_frame={reference_frame} "
            f"({ref_wilor} missing). Try a different --reference_frame."
        )
    with open(ref_wilor) as f:
        detections = json.load(f)
    if not detections:
        raise RuntimeError(
            f"WiLoR found no hands in reference frame {reference_frame}. "
            f"Try a different --reference_frame."
        )
    # Cache the ref-frame slice next to other prompt artifacts so it can be
    # re-loaded by hand by anyone debugging.
    with open(hand_detections, "w") as f:
        json.dump(detections, f, indent=2)

    if sam2_bbox_pad > 0.0:
        ref_w, ref_h = Image.open(ref_rgb).size
        for det in detections:
            det["bbox"] = _pad_bbox(
                det["bbox"],
                sam2_bbox_pad,
                ref_w,
                ref_h,
                square=sam2_square_wilor_bboxes,
            )

    # 4. Object detection (optional) -----------------------------------------
    object_track_id: int | None = None
    object_box = None
    obj_dets: list[dict] = []
    if object_prompt is not None:
        if not _step("Grounding DINO (ref frame)", os.path.exists(dino_detections)):
            run_image_to_object_bboxes(
                image_path  = ref_rgb,
                output_path = dino_detections,
                prompt      = object_prompt,
                model_dir   = grounding_dino_weights,
                dev         = False,
            )
        with open(dino_detections) as f:
            obj_dets = json.load(f)
        if not obj_dets:
            raise RuntimeError(
                f"Grounding DINO found nothing for prompt {object_prompt!r}. "
                f"Try a different prompt or --reference_frame."
            )
        object_box = BoundingBox.from_dict(obj_dets[0]["box"])
        object_track_id = 1

    # 5. Build SAM2 prompts --------------------------------------------------
    # Object prompts remain bbox-driven. Hand prompts can either use the
    # WiLoR bbox (legacy/default) or a rendered WiLoR MANO silhouette mask
    # on the reference frame. SAM2 consumes the mask as its conditioning
    # prompt and then propagates it through the video.
    rebuilt_sam2_prompts = False
    hand_id_offset = 1 if object_track_id is None else 2
    if not _sam2_prompt_artifacts_match(
        sam2_prompts,
        hand_tracks,
        sam2_hand_prompt_source,
    ):
        prompts: list[Sam2Prompt] = []
        hand_meta: list[dict] = []
        if object_track_id is not None:
            prompts.append(Sam2Prompt(
                frame_index=reference_frame,
                object_id=object_track_id,
                box=object_box,
            ))

        if sam2_hand_prompt_source == "wilor_mask":
            expected_masks = [
                os.path.join(
                    sam2_hand_prompt_masks_dir,
                    str(i + hand_id_offset),
                    f"{reference_frame:06d}.png",
                )
                for i in range(len(detections))
            ]
            if not _step("Render WiLoR hand-mask prompts",
                         all(os.path.exists(p) for p in expected_masks)):
                run_render_wilor_prompt_masks(
                    wilor_json       = ref_wilor,
                    image_path       = ref_rgb,
                    output_dir       = sam2_hand_prompt_masks_dir,
                    mano_assets_root = mano_assets_root,
                    frame_index      = reference_frame,
                    first_object_id  = hand_id_offset,
                    dev              = dev,
                )

        for i, det in enumerate(detections):
            oid = i + hand_id_offset
            prompt_kwargs = {
                "frame_index": reference_frame,
                "object_id": oid,
            }
            track_record = {
                "object_id": oid,
                "role":      "hand",
                "is_right":  bool(det["is_right"]),
                "score":     float(det["score"]),
            }
            if sam2_hand_prompt_source == "wilor_mask":
                mask_path = os.path.join(
                    sam2_hand_prompt_masks_dir,
                    str(oid),
                    f"{reference_frame:06d}.png",
                )
                if not os.path.exists(mask_path):
                    raise RuntimeError(f"Missing rendered WiLoR prompt mask: {mask_path}")
                rel_mask_path = os.path.relpath(
                    mask_path,
                    os.path.dirname(os.path.abspath(sam2_prompts)),
                )
                prompt_kwargs["mask_path"] = rel_mask_path
                track_record["sam2_prompt_mask_path"] = rel_mask_path
            else:
                prompt_kwargs["box"] = BoundingBox.from_dict(det["bbox"])
            prompts.append(Sam2Prompt(**prompt_kwargs))
            hand_meta.append(track_record)

        Sam2Prompts(prompts=prompts).save(sam2_prompts)
        with open(hand_tracks, "w") as f:
            json.dump({
                "reference_frame": reference_frame,
                "sam2_hand_prompt_source": sam2_hand_prompt_source,
                "tracks":          hand_meta,
            }, f, indent=2)
        rebuilt_sam2_prompts = True
        print(f"  Wrote {len(prompts)} prompt(s) → {sam2_prompts}")
        print(f"  Wrote hand track metadata → {hand_tracks}")
        if object_track_id is not None:
            with open(object_track, "w") as f:
                json.dump({
                    "reference_frame": reference_frame,
                    "object_id":       object_track_id,
                    "prompt":          object_prompt,
                    "label":           obj_dets[0].get("label"),
                    "confidence":      float(obj_dets[0].get("confidence", 0.0)),
                }, f, indent=2)
            print(f"  Wrote object track metadata → {object_track}")

    # 6. SAM2 propagation ----------------------------------------------------
    sam2_done = (not rebuilt_sam2_prompts) and any(
        _has_files(os.path.join(masks_dir, d))
        for d in (os.listdir(masks_dir) if os.path.isdir(masks_dir) else [])
    )
    if not _step("SAM2 mask propagation", sam2_done):
        run_video_to_masks(
            video_path  = video_path,
            prompts_path= sam2_prompts,
            masks_dir   = masks_dir,
            weights_dir = sam2_weights,
            dev         = dev,
        )

    # 7. Verification renders ------------------------------------------------
    with open(hand_tracks) as f:
        track_meta = json.load(f)["tracks"]

    if not _step("Render prompts overlay", os.path.exists(prompts_overlay)):
        _render_prompts_overlay(ref_rgb, track_meta, detections, prompts_overlay)

    if not _step("Render masks overlay video", os.path.exists(masks_overlay)):
        _render_masks_overlay(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            tracks=track_meta,
            output_path=masks_overlay,
        )

    # 8. Object branch: mesh source + FoundationPose -------------------------
    if object_prompt is not None:
        ref_obj_mask = f"{masks_dir}/{object_track_id}/{reference_frame:06d}.png"
        if object_mesh_path is not None:
            object_mesh_for_pipeline = object_mesh_path
            print("  [skip] SAM3D mesh generation (using provided OBJ)")
            print("  [skip] FoundationPose scale estimation (using provided OBJ scale)")
        else:
            os.makedirs(mesh_dir, exist_ok=True)
            if not _step("SAM3D mesh generation", os.path.exists(mesh_path)):
                run_image_to_mesh(
                    image_path            = ref_rgb,
                    mask_path             = ref_obj_mask,
                    mesh_path             = mesh_path,
                    transform_path        = mesh_transform,
                    intrinsics_path       = mesh_intrinsics,
                    weights_dir           = sam3d_weights,
                    with_texture_baking   = True,
                    with_mesh_postprocess = True,
                    depth_path            = ref_depth,
                    depth_intrinsics_path = intrinsics_stable,
                    depth_mask_path       = ref_obj_mask,
                    dev                   = dev,
                )

            if not _step("Apply SAM3D transform", os.path.exists(mesh_pretransformed)):
                _apply_sam3d_transform(mesh_path, mesh_transform, mesh_pretransformed)

            if not _step("FoundationPose scale estimation", os.path.exists(mesh_scaled)):
                run_estimate_mesh_scale(
                    mesh_path               = mesh_pretransformed,
                    rgb_path                = ref_rgb,
                    depth_path              = ref_depth,
                    mask_path               = ref_obj_mask,
                    intrinsics_path         = intrinsics_stable,
                    weights_dir             = foundation_pose_weights,
                    scale_path              = scale_path,
                    rescaled_mesh_path      = mesh_scaled,
                    lo                      = 0.5,
                    hi                      = 2.0,
                    n_samples               = 9,
                    n_levels                = 4,
                    iou_weight              = 1.0,
                    depth_weight            = 1.0,
                    registration_iterations = 5,
                    dev                     = dev,
                )
            object_mesh_for_pipeline = mesh_scaled

        if not _step("FoundationPose tracking", _has_files(poses_dir)):
            run_video_to_poses(
                video_path             = video_path,
                depth_folder           = depth_dir,
                masks_folder           = f"{masks_dir}/{object_track_id}",
                camera_intrinsics_path = intrinsics_stable,
                mesh_path              = object_mesh_for_pipeline,
                poses_dir              = poses_dir,
                weights_dir            = foundation_pose_weights,
                reference_frame        = reference_frame,
                mask_depth             = True,
                reregister_iou_thresh  = reregister_iou_thresh if reregister_iou_thresh else None,
                dev                    = dev,
            )

        if not _step("EKF pose smoothing", _has_files(poses_smooth_dir)):
            run_ekf_smoothing(
                poses_dir            = poses_dir,
                mesh_path            = object_mesh_for_pipeline,
                intrinsics_path      = intrinsics_stable,
                weights_dir          = foundation_pose_weights,
                output_dir           = poses_smooth_dir,
                masks_folder         = f"{masks_dir}/{object_track_id}",
                process_noise_xy     = 0.01,
                process_noise_z      = 0.01,
                process_noise_r      = 0.02,
                measurement_noise_xy = 0.01,
                measurement_noise_z  = 0.04,
                measurement_noise_r  = 0.02,
                dev                  = dev,
            )

    # Downstream object overlays/refinement use the object mask from masks/.
    object_masks_dir = (
        f"{masks_dir}/{object_track_id}"
        if (object_prompt is not None and object_track_id is not None) else None
    )
    object_mesh_arg  = object_mesh_for_pipeline if (object_prompt is not None) else None
    object_poses_arg = poses_smooth_dir          if (object_prompt is not None) else None

    # 9. IoU-match wilor detections to SAM2 hand tracks ----------------------
    if not use_hamer_primary:
        matching_done = any(
            _has_files(os.path.join(wilor_tracks_dir, d))
            for d in (os.listdir(wilor_tracks_dir) if os.path.isdir(wilor_tracks_dir) else [])
        )
        if not _step("Match wilor detections → SAM2 tracks (silhouette IoU)",
                     matching_done):
            run_tracks_from_wilor_masks(
                frames_dir       = frames_dir,
                wilor_raw_dir    = wilor_raw_dir,
                masks_dir        = masks_dir,
                tracks_path      = hand_tracks,
                output_dir       = wilor_tracks_dir,
                mano_assets_root = mano_assets_root,
                min_iou          = min_iou,
                dev              = dev,
            )
    else:
        print("  [skip] Match wilor detections -> SAM2 tracks (using HaMeR from SAM2 masks)")

    # 9b. (When refining masks) Fill wilor track gaps BEFORE alignment so
    # mask refinement gets a record for every visible frame (SAM2 mask
    # present). This pre-align fill is gated on the same flag — only when
    # we're going to consume the filled records downstream (mask refine).
    # Alignment itself still reads from wilor/ (real only), since aligning a
    # guess against real depth would corrupt cam_t.
    if refine_masks_with_silhouette:
        wilor_filled_done = any(
            _has_files(os.path.join(wilor_filled_dir, d))
            for d in (os.listdir(wilor_filled_dir)
                      if os.path.isdir(wilor_filled_dir) else [])
        )
        if not _step("Interpolate wilor tracks (pre-align)", wilor_filled_done):
            run_wilor_tracks_interpolate(
                wilor_dir      = wilor_tracks_dir,
                masks_dir      = masks_dir,
                output_dir     = wilor_filled_dir,
                betas          = interp_betas,
                max_gap_frames = interp_max_gap_frames,
                dev            = dev,
            )

    # 9c. Refine SAM2 hand masks by intersecting with a dilated rendered
    # MANO silhouette. Reads from wilor_filled_dir so every frame with a
    # SAM2 mask (within [first_real, last_real] per track) gets a refined
    # mask. Object mask is left untouched in masks/ (we only have wilor
    # silhouettes for hands).
    if refine_masks_with_silhouette:
        refined_done = any(
            _has_files(os.path.join(masks_refined_dir, d))
            for d in (os.listdir(masks_refined_dir)
                      if os.path.isdir(masks_refined_dir) else [])
        )
        if not _step("Refine SAM2 hand masks via MANO silhouette ∩",
                     refined_done):
            run_masks_intersect_silhouette(
                wilor_dir        = wilor_filled_dir,
                masks_dir        = masks_dir,
                tracks_path      = hand_tracks,
                output_dir       = masks_refined_dir,
                mano_assets_root = mano_assets_root,
                dilation_pixels  = mask_dilation_pixels,
                dev              = dev,
            )

        # Verification overlay over the refined masks (hand tracks only —
        # object stays in masks/, gets rendered by the existing overlay).
        if not _step("Render refined masks overlay",
                     os.path.exists(masks_refined_overlay)):
            _render_masks_overlay(
                frames_dir = frames_dir,
                masks_dir  = masks_refined_dir,
                tracks     = track_meta,
                output_path= masks_refined_overlay,
            )

    # Downstream steps source hand masks from `hand_masks_dir`; object masks
    # always come from `masks_dir`.
    hand_masks_dir = masks_refined_dir if refine_masks_with_silhouette else masks_dir

    if not use_hamer_primary:
        # 10. Render virtual-cam verification overlay (raw, real detections only)
        if not _step("Render wilor mesh overlay (raw)", os.path.exists(wilor_overlay)):
            run_wilor_render(
                frames_dir       = frames_dir,
                wilor_dir        = wilor_tracks_dir,
                mano_assets_root = mano_assets_root,
                output_path      = wilor_overlay,
                dev              = dev,
            )

        # 11. Depth alignment to MoGe + real intrinsics (real frames only) --
        # Run alignment on the real-only matched detections. Interpolated frames
        # would lie on a guessed pose that may not match the image silhouette, so
        # aligning them against real depth would corrupt cam_t. We interpolate
        # AFTER alignment (step 13).
        # Reuses hamer's run_align_hands — schema is hamer-compatible.
        if not _step("Align wilor hands to depth (real frames only)", False):
            run_align_hands(
                hamer_dir         = wilor_tracks_dir,
                depth_dir         = depth_dir,
                intrinsics_path   = intrinsics_stable,
                mano_assets_root  = mano_assets_root,
                output_dir        = wilor_aligned_dir,
                hand_masks_dir    = hand_masks_dir,
                object_masks_dir  = object_masks_dir,
                mask_min_pixels   = mask_min_pixels,
                dev               = dev,
            )

        # 12. Render aligned overlay (real frames only) ---------------------
        if not _step("Render aligned wilor overlay (real only)",
                     os.path.exists(wilor_aligned_overlay)):
            run_render_hands_aligned_video(
                frames_dir         = frames_dir,
                aligned_dir        = wilor_aligned_dir,
                mano_assets_root   = mano_assets_root,
                output_path        = wilor_aligned_overlay,
                object_mesh_path   = object_mesh_arg,
                object_poses_dir   = object_poses_arg,
                dev                = dev,
            )

    # 13. Interpolate per-track gaps on the ALIGNED records ----------------
    # Real aligned frames pass through (interpolated=false). Missing frames
    # within --interp_max_gap_frames get SLERP/linear-filled (interpolated=
    # true), gated on SAM2 mask presence (only fill where the hand is
    # actually visible). `betas` policy per --interp_betas.
    if not use_hamer_primary:
        aligned_filled_done = any(
            _has_files(os.path.join(wilor_aligned_filled_dir, d))
            for d in (os.listdir(wilor_aligned_filled_dir)
                      if os.path.isdir(wilor_aligned_filled_dir) else [])
        )
        if not _step("Interpolate missing aligned frames", aligned_filled_done):
            run_tracks_interpolate(
                aligned_dir    = wilor_aligned_dir,
                masks_dir      = hand_masks_dir,
                output_dir     = wilor_aligned_filled_dir,
                betas          = interp_betas,
                max_gap_frames = interp_max_gap_frames,
                dev            = dev,
            )

        # 13b. Reconcile hand_tracks.json with canonical handedness emitted by
        # interpolation (whole-sequence majority vote + position tiebreak).
        _reconcile_hand_tracks_from_handedness(
            hand_tracks,
            f"{wilor_aligned_filled_dir}/handedness.json",
        )

        # 14. Render filled aligned overlay (real + interpolated) ----------
        if not _step("Render filled aligned wilor overlay",
                     os.path.exists(wilor_aligned_filled_overlay)):
            run_render_hands_aligned_video(
                frames_dir         = frames_dir,
                aligned_dir        = wilor_aligned_filled_dir,
                mano_assets_root   = mano_assets_root,
                output_path        = wilor_aligned_filled_overlay,
                object_mesh_path   = object_mesh_arg,
                object_poses_dir   = object_poses_arg,
                dev                = dev,
            )

    # 14a–14e. Optional/primary HaMeR pass. Runs HaMeR per-track per-frame
    # using SAM2 masks and the handedness from hand_tracks.json, then aligns
    # to depth, interpolates, and renders. In --hand_pose_source hamer mode,
    # this is the primary hand track source.
    hamer_mano_assets_root = os.path.join(hamer_weights, "_DATA", "data")
    if run_hamer_outputs:
        # 14a. HaMeR per-track regression on the hand masks.
        if not _step("HaMeR per-frame regression", False):
            run_hamer_masks_to_hands(
                frames_dir       = frames_dir,
                masks_dir        = hand_masks_dir,
                tracks_path      = hand_tracks,
                output_dir       = hamer_dir,
                weights_dir      = hamer_weights,
                bbox_expansion   = hamer_bbox_expansion,
                mask_min_pixels  = hamer_mask_min_pixels,
                dev              = dev,
            )

        # 14b. HaMeR virtual-cam overlay.
        if not _step("Render HaMeR mesh overlay", os.path.exists(hamer_overlay)):
            run_hamer_render(
                frames_dir       = frames_dir,
                hamer_dir        = hamer_dir,
                mano_assets_root = hamer_mano_assets_root,
                output_path      = hamer_overlay,
                dev              = dev,
            )

        # 14c. Align HaMeR to depth.
        if not _step("Align HaMeR hands to depth", False):
            run_align_hands(
                hamer_dir         = hamer_dir,
                depth_dir         = depth_dir,
                intrinsics_path   = intrinsics_stable,
                mano_assets_root  = hamer_mano_assets_root,
                output_dir        = hamer_aligned_dir,
                hand_masks_dir    = hand_masks_dir,
                object_masks_dir  = object_masks_dir,
                mask_min_pixels   = hamer_mask_min_pixels,
                dev               = dev,
            )

        # 14d. HaMeR aligned overlay.
        if not _step("Render aligned HaMeR overlay",
                     os.path.exists(hamer_aligned_overlay)):
            run_render_hands_aligned_video(
                frames_dir         = frames_dir,
                aligned_dir        = hamer_aligned_dir,
                mano_assets_root   = hamer_mano_assets_root,
                output_path        = hamer_aligned_overlay,
                object_mesh_path   = object_mesh_arg,
                object_poses_dir   = object_poses_arg,
                dev                = dev,
            )

        # 14e. Interpolate HaMeR aligned tracks (reuses tracks_interpolate;
        # also canonicalizes handedness within the hamer-side records).
        hamer_filled_done = any(
            _has_files(os.path.join(hamer_aligned_filled_dir, d))
            for d in (os.listdir(hamer_aligned_filled_dir)
                      if os.path.isdir(hamer_aligned_filled_dir) else [])
        )
        if not _step("Interpolate HaMeR aligned frames", hamer_filled_done):
            run_tracks_interpolate(
                aligned_dir    = hamer_aligned_dir,
                masks_dir      = hand_masks_dir,
                output_dir     = hamer_aligned_filled_dir,
                betas          = interp_betas,
                max_gap_frames = interp_max_gap_frames,
                dev            = dev,
            )
        _reconcile_hand_tracks_from_handedness(
            hand_tracks,
            f"{hamer_aligned_filled_dir}/handedness.json",
        )

        # 14i. Filled HaMeR aligned overlay.
        if not _step("Render filled aligned HaMeR overlay",
                     os.path.exists(hamer_aligned_filled_overlay)):
            run_render_hands_aligned_video(
                frames_dir         = frames_dir,
                aligned_dir        = hamer_aligned_filled_dir,
                mano_assets_root   = hamer_mano_assets_root,
                output_path        = hamer_aligned_filled_overlay,
                object_mesh_path   = object_mesh_arg,
                object_poses_dir   = object_poses_arg,
                dev                = dev,
            )

    # 15. Optional joint hand+object refinement via Gaussian splatting -----
    # Mirrors run_hand_masks.py's refinement block. The selected primary hand
    # source supplies aligned+filled records, with one pose per visible mask
    # frame after interpolation.
    refine_source              = "hamer" if run_hamer_outputs else "wilor"
    refine_hand_pose_dir       = hamer_aligned_filled_dir if run_hamer_outputs else wilor_aligned_filled_dir
    refine_hand_masks_dir      = hand_masks_dir
    refine_object_mask_dir     = object_masks_dir
    refine_mano_assets         = hamer_mano_assets_root if run_hamer_outputs else mano_assets_root

    refined_poses_dir          = f"{output_dir}/poses_refined"
    refined_hand_dir           = f"{output_dir}/{refine_source}_refined"
    refined_overlay            = f"{output_dir}/refined_overlay.mp4"
    refined_hand_overlay       = f"{output_dir}/{refine_source}_refined_overlay.mp4"
    refined_object_scale_json  = f"{output_dir}/refined_object_scale.json"
    refine_checkpoint_pt       = f"{output_dir}/refine_checkpoint.pt"
    refined_overlay_pass2      = f"{output_dir}/refined_overlay_pass2.mp4"
    refine_checkpoint_pt_pass2 = f"{output_dir}/refine_checkpoint_pass2.pt"
    refined_simple_poses_dir   = f"{output_dir}/poses_refined_simple"
    refined_simple_hand_dir    = f"{output_dir}/{refine_source}_refined_simple"
    refined_simple_camera_dir  = f"{output_dir}/camera_refined_simple"
    refined_simple_overlay     = f"{output_dir}/refined_simple_overlay.mp4"
    refined_simple_hand_overlay = f"{output_dir}/{refine_source}_refined_simple_overlay.mp4"
    refined_simple_progress_dir = f"{output_dir}/refined_simple_progress"
    refined_simple_progress_video = f"{output_dir}/progress_simple.mp4"
    if run_refinement:
        if object_prompt is None:
            raise ValueError(
                "--run_refinement requires --object_prompt (joint hand+object refinement)."
            )

        with open(hand_tracks) as f:
            htm = json.load(f)["tracks"]
        left_candidates  = [t["object_id"] for t in htm if not t["is_right"]]
        right_candidates = [t["object_id"] for t in htm if     t["is_right"]]
        left_tid: int | None  = left_candidates[0]  if left_candidates  else None
        right_tid: int | None = right_candidates[0] if right_candidates else None
        # Silent track-drop has bitten us before — surface ambiguities loudly.
        if len(left_candidates) > 1 or len(right_candidates) > 1:
            print("  WARNING: hand_tracks.json has duplicate-handedness tracks; "
                  f"refinement will only refine one per side.\n"
                  f"            left  candidates = {left_candidates}\n"
                  f"            right candidates = {right_candidates}\n"
                  f"            using left_tid={left_tid}, right_tid={right_tid}.\n"
                  f"            (Inspect {refine_hand_pose_dir}/handedness.json "
                  f"and consider re-running with corrected handedness.)")
        if left_tid is None and right_tid is None:
            raise RuntimeError(
                f"No hand tracks found in {hand_tracks}; cannot run refinement."
            )
        for t in htm:
            if t["object_id"] not in (left_tid, right_tid):
                print(f"  WARNING: track {t['object_id']} (is_right="
                      f"{t['is_right']}) will be DROPPED by gsplat refinement "
                      f"(only one track per handedness is consumed).")

        def _maybe(d: str | None) -> str | None:
            return d if d is not None else None

        left_pose_in   = f"{refine_hand_pose_dir}/{left_tid}"  if left_tid  is not None else None
        right_pose_in  = f"{refine_hand_pose_dir}/{right_tid}" if right_tid is not None else None
        left_mask_in   = f"{refine_hand_masks_dir}/{left_tid}"  if left_tid  is not None else None
        right_mask_in  = f"{refine_hand_masks_dir}/{right_tid}" if right_tid is not None else None
        left_pose_out  = f"{refined_hand_dir}/{left_tid}"      if left_tid  is not None else None
        right_pose_out = f"{refined_hand_dir}/{right_tid}"     if right_tid is not None else None
        print(f"  Refinement source: {refine_source} ({refine_hand_pose_dir}, "
              f"hand masks: {refine_hand_masks_dir}, "
              f"object mask: {refine_object_mask_dir})")

        pass1_kwargs: dict = dict(
                frames_dir                  = frames_dir,
                intrinsics_path             = intrinsics_stable,
                object_mesh_path            = object_mesh_for_pipeline,
                object_poses_dir            = poses_dir,
                object_mask_dir             = refine_object_mask_dir,
                refined_object_poses_dir    = refined_poses_dir,
                overlay_path                = refined_overlay,
                refined_object_scale_path   = None if using_provided_object_mesh else refined_object_scale_json,
                checkpoint_path             = refine_checkpoint_pt,
                resume_from_checkpoint      = (
                    refine_checkpoint_pt
                    if refinement_resume and os.path.exists(refine_checkpoint_pt)
                    else None
                ),
                freeze_gaussians            = refinement_object_pose_only,
                freeze_hand_rot             = refinement_object_pose_only,
                freeze_hand_trans           = refinement_object_pose_only,
                freeze_bg_rot               = refinement_object_pose_only,
                freeze_bg_trans             = refinement_object_pose_only,
                random_init_obj_pose        = refinement_random_init_obj_pose,
                ignore_optimizer_state      = refinement_ignore_optimizer_state,
                left_hand_pose_dir          = _maybe(left_pose_in),
                left_hand_mask_dir          = _maybe(left_mask_in),
                right_hand_pose_dir         = _maybe(right_pose_in),
                right_hand_mask_dir         = _maybe(right_mask_in),
                refined_left_hand_pose_dir  = _maybe(left_pose_out),
                refined_right_hand_pose_dir = _maybe(right_pose_out),
                depth_dir                   = depth_dir,
                mano_assets_root            = refine_mano_assets,
                n_epochs                    = 32,
                batch_size                  = 16,
                render_every                = 25,
                lr_gaussians                = 1e-4,
                lr_hand_gaussians           = 1e-4,
                lr_mul_delta_p              = 10.0,
                lr_mul_quat                 = 0.5,
                lr_mul_scale                = 3.0,
                lr_mul_opacity              = 3.0,
                lr_mul_color                = 1.0,
                lr_mul_obj_global_scale     = 0.0 if using_provided_object_mesh else 1.0,
                lr_object_pose              = 1e-3,
                lr_object_rot               = 1e-2,
                lr_object_trans             = 1e-3,
                lr_hand_pose                = 1e-3,
                lr_hand_global_orient       = 1e-2,
                lr_hand_finger              = 1e-4,
                lr_hand_trans               = 1e-3,
                lr_betas                    = 1e-2,
                learn_hand_scale            = True,
                lr_hand_scale               = 1.0,
                w_hand_scale_prior          = 1.0,
                w_photometric               = 1.0,
                w_silhouette                = 1.0,
                w_silhouette_hand           = 1.0,
                w_silhouette_obj            = 1.0,
                w_depth                     = 1.0,
                w_log_depth_grad            = 10.0,
                w_photometric_ssim          = 10.0,
                w_depth_ssim                = 10.0,
                w_delta_p_reg_obj           = 0.01,
                w_delta_p_reg_hand          = 0.01,
                w_delta_p_reg_bg            = 0.000,
                w_smooth_obj_rot            = w_smooth_obj_rot,
                w_smooth_obj_trans          = w_smooth_obj_trans,
                w_smooth_hand_rot           = w_smooth_hand_rot,
                w_smooth_hand_finger        = w_smooth_hand_finger,
                w_smooth_hand_trans         = w_smooth_hand_trans,
                w_smooth_bg_rot             = w_smooth_bg_rot,
                w_smooth_bg_trans           = w_smooth_bg_trans,
                w_beta_prior                = 1.0,
                w_obj_scale_prior           = 0.0 if using_provided_object_mesh else 1.0,
                freeze_object_scale         = using_provided_object_mesh or refinement_object_pose_only,
                n_gaussian_only_epochs      = 0,
                seed                        = 0,
                with_background             = True,
                background_pose_init_dir    = slam_poses_dir if run_slam else None,
                mask_background_to_black    = False,
                balance_photometric_by_mask = False,
                bg_ref_frame                = reference_frame,
                lr_bg_gaussians             = 1e-4,
                lr_bg_pose                  = 1e-4,
                lr_bg_rot                   = 1e-4,
                lr_bg_trans                 = 1e-4,
                use_cosine_lr_schedule      = True,
                cosine_lr_min_ratio         = 0.1,
                coarse_init_scale_factor    = 1.0,
                pose_confidence_decay       = 0.0,
                pose_confidence_ref_frame   = reference_frame,
                w_pose_init_prior           = 0.1,
                rotation_search_n_candidates      = 0,
                rotation_search_period            = 0,
                rotation_search_local_frac        = 0.5,
                rotation_search_local_max_deg     = 15.0,
                rotation_search_silhouette_weight = 1.0,
                rotation_search_smoothness_weight = rotation_search_smoothness_weight,
                use_l2_photometric                = True,
                use_l2_silhouette                 = True,
                pose_confidence_dynamic_tau       = 0.0,
                train_resolution_scale            = 0.5,
                n_obj_gaussians                   = 5000,
                bg_max_points                     = bg_max_points,
                bg_init_stride                    = bg_init_stride,
                bg_voxel_size                     = bg_voxel_size,
                w_scale_aniso_bg                  = 0.1,
                w_density_bg                      = 0.0,
                n_density_neighbors               = n_density_neighbors,
                density_subsample_frac_bg         = density_subsample_frac_bg,
                w_sdf_density_bg                  = 0.0,
                w_normal_consistency_bg           = 0.0,
                n_sdf_samples_bg                  = n_sdf_samples_bg,
                n_sdf_neighbors_bg                = n_sdf_neighbors_bg,
                valid_mask_threshold              = valid_mask_threshold,
                valid_mask_erode_iters            = valid_mask_erode_iters,
                dev                         = dev,
                hand_anchor_mode            = "face",
                w_face_delta_p_normal_outward_hand  = 100.0,
                w_face_delta_p_normal_inward_hand   = 100.0,
                w_face_delta_p_tangent_hand         = 100.0,
                object_anchor_mode            = "face",
                face_normal_thin_factor_obj   = 0.25,
                init_opacity_obj            = init_opacity_obj_eff,
                init_opacity_hand           = init_opacity_hand_eff,
                init_opacity_bg             = init_opacity_bg_eff,
                init_opacity_wrist          = init_opacity_wrist_eff,
                init_gaussian_scale_factor  = init_gaussian_scale_factor,
                checkpoint_every            = 25,  
                w_opacity_binary_hand       = 1.0,
                w_opacity_binary_obj        = 1.0,
                w_opacity_binary_bg         = 1.0,
                w_depth_variance            = 1.0,
                w_depth_ordering            = 1.0,
                depth_ordering_margin       = 0.0,
                smooth_obj_in_world         = smooth_obj_in_world,
                smooth_hand_in_world        = smooth_hand_in_world,
                n_wrist_gaussians           = 15,
                learn_focal                 = True,
                learn_principal_point       = True,
                w_intrinsics_prior          = 1e3,
                snap_rotation_outliers_every = 5,
                snap_rotation_targets = "obj,hand_wrist",
                snap_rotation_threshold = 0.5,
                snap_rotation_verbose = True,
                snap_rotation_window = 51,
        )
        if not _step("Gaussian-splat refinement (pass 1)",
                     os.path.exists(refined_overlay)):
            run_refine(**pass1_kwargs)

        if refinement_second_pass:
            sp_lr = float(refinement_second_pass_lr_scale)
            sp_n  = (refinement_second_pass_epochs
                     if refinement_second_pass_epochs is not None
                     else max(8, pass1_kwargs["n_epochs"] // 2))
            sp_bs = (refinement_second_pass_batch_size
                     if refinement_second_pass_batch_size is not None
                     else pass1_kwargs["batch_size"] * 4)
            pass2_kwargs = {
                **pass1_kwargs,
                "n_epochs":                sp_n,
                "batch_size":              sp_bs,
                "n_gaussian_only_epochs":  0,
                "checkpoint_path":         refine_checkpoint_pt_pass2,
                "resume_from_checkpoint":  refine_checkpoint_pt,
                "ignore_optimizer_state":  True,
                "overlay_path":            refined_overlay_pass2,
                "random_init_obj_pose":    False,
                "lr_gaussians":          pass1_kwargs["lr_gaussians"]          * sp_lr,
                "lr_hand_gaussians":     pass1_kwargs["lr_hand_gaussians"]     * sp_lr,
                "lr_object_rot":         pass1_kwargs["lr_object_rot"]         * sp_lr,
                "lr_object_trans":       pass1_kwargs["lr_object_trans"]       * sp_lr,
                "lr_hand_global_orient": pass1_kwargs["lr_hand_global_orient"] * sp_lr,
                "lr_hand_finger":        pass1_kwargs["lr_hand_finger"]        * sp_lr,
                "lr_hand_trans":         pass1_kwargs["lr_hand_trans"]         * sp_lr,
                "lr_bg_gaussians":       pass1_kwargs["lr_bg_gaussians"]       * sp_lr,
                "lr_bg_rot":             pass1_kwargs["lr_bg_rot"]             * sp_lr,
                "lr_bg_trans":           pass1_kwargs["lr_bg_trans"]           * sp_lr,
                "lr_betas":              pass1_kwargs["lr_betas"]              * sp_lr,
            }
            if not _step("Gaussian-splat refinement (pass 2)",
                         os.path.exists(refined_overlay_pass2)):
                run_refine(**pass2_kwargs)

        # Render refined hands + object using the aligned-overlay renderer
        # so it's directly comparable to wilor_aligned_overlay.mp4.
        learned_object_scale = 1.0
        if (not using_provided_object_mesh) and os.path.exists(refined_object_scale_json):
            with open(refined_object_scale_json) as f:
                learned_object_scale = float(json.load(f).get("scale", 1.0))
            print(f"  Loaded learned object scale: {learned_object_scale:.4f}")
        if not _step(f"Render refined {refine_source} overlay",
                     os.path.exists(refined_hand_overlay)):
            run_render_hands_aligned_video(
                frames_dir         = frames_dir,
                aligned_dir        = refined_hand_dir,
                mano_assets_root   = refine_mano_assets,
                output_path        = refined_hand_overlay,
                object_mesh_path   = object_mesh_for_pipeline,
                object_poses_dir   = refined_poses_dir,
                object_scale       = learned_object_scale,
                dev                = dev,
            )

        _plot_pose_grad_from_checkpoint(
            refine_checkpoint_pt,
            f"{output_dir}/pose_grad.png",
        )

    if run_refinement_simple:
        if object_prompt is None:
            raise ValueError(
                "--run_refinement_simple requires --object_prompt (hand+object refinement)."
            )
        if object_mesh_for_pipeline is None:
            raise RuntimeError(
                "--run_refinement_simple requires an object mesh from the object branch."
            )

        with open(hand_tracks) as f:
            htm = json.load(f)["tracks"]
        left_candidates  = [t["object_id"] for t in htm if not t["is_right"]]
        right_candidates = [t["object_id"] for t in htm if     t["is_right"]]
        left_tid: int | None  = left_candidates[0]  if left_candidates  else None
        right_tid: int | None = right_candidates[0] if right_candidates else None
        if len(left_candidates) > 1 or len(right_candidates) > 1:
            print("  WARNING: hand_tracks.json has duplicate-handedness tracks; "
                  f"simple refinement will only refine one per side.\n"
                  f"            left  candidates = {left_candidates}\n"
                  f"            right candidates = {right_candidates}\n"
                  f"            using left_tid={left_tid}, right_tid={right_tid}.")
        if left_tid is None and right_tid is None:
            raise RuntimeError(
                f"No hand tracks found in {hand_tracks}; cannot run simple refinement."
            )
        for t in htm:
            if t["object_id"] not in (left_tid, right_tid):
                print(f"  WARNING: track {t['object_id']} (is_right="
                      f"{t['is_right']}) will be DROPPED by simple refinement "
                      f"(only one track per handedness is consumed).")

        def _maybe_simple(d: str | None) -> str | None:
            return d if d is not None else None

        left_pose_in   = f"{refine_hand_pose_dir}/{left_tid}"  if left_tid  is not None else None
        right_pose_in  = f"{refine_hand_pose_dir}/{right_tid}" if right_tid is not None else None
        left_mask_in   = f"{refine_hand_masks_dir}/{left_tid}"  if left_tid  is not None else None
        right_mask_in  = f"{refine_hand_masks_dir}/{right_tid}" if right_tid is not None else None
        left_pose_out  = f"{refined_simple_hand_dir}/{left_tid}"      if left_tid  is not None else None
        right_pose_out = f"{refined_simple_hand_dir}/{right_tid}"     if right_tid is not None else None
        print(f"  Simple refinement source: {refine_source} ({refine_hand_pose_dir}, "
              f"hand masks: {refine_hand_masks_dir}, "
              f"object mask: {refine_object_mask_dir})")

        if not _step("Simple 2DGS refinement", os.path.exists(refined_simple_overlay)):
            run_refine_simple(
                frames_dir                  = frames_dir,
                depth_dir                   = depth_dir,
                intrinsics_path             = intrinsics_stable,
                object_mesh_path            = object_mesh_for_pipeline,
                object_poses_dir            = poses_dir,
                object_mask_dir             = refine_object_mask_dir,
                refined_object_poses_dir    = refined_simple_poses_dir,
                overlay_path                = refined_simple_overlay,
                left_hand_pose_dir          = _maybe_simple(left_pose_in),
                left_hand_mask_dir          = _maybe_simple(left_mask_in),
                right_hand_pose_dir         = _maybe_simple(right_pose_in),
                right_hand_mask_dir         = _maybe_simple(right_mask_in),
                refined_left_hand_pose_dir  = _maybe_simple(left_pose_out),
                refined_right_hand_pose_dir = _maybe_simple(right_pose_out),
                refined_camera_poses_dir    = refined_simple_camera_dir,
                mano_assets_root            = refine_mano_assets,
                background_pose_init_dir    = slam_poses_dir if run_slam else None,
                n_epochs                    = simple_refinement_epochs,
                batch_size                  = simple_refinement_batch_size,
                train_resolution_scale      = simple_refinement_train_resolution_scale,
                lr_gaussians                = simple_refinement_lr_gaussians,
                lr_object_pose              = simple_refinement_lr_object_pose,
                lr_object_scale             = simple_refinement_lr_object_scale,
                lr_hand_pose                = simple_refinement_lr_hand_pose,
                lr_hand_articulation        = simple_refinement_lr_hand_articulation,
                lr_hand_shape               = simple_refinement_lr_hand_shape,
                lr_hand_scale               = simple_refinement_lr_hand_scale,
                lr_camera_pose              = simple_refinement_lr_camera_pose,
                lr_schedule                 = simple_refinement_lr_schedule,
                lr_cosine_min_factor        = simple_refinement_lr_cosine_min_factor,
                w_smooth_object_rot         = simple_refinement_w_smooth_object_rot,
                w_smooth_object_trans       = simple_refinement_w_smooth_object_trans,
                w_smooth_hand_rot           = simple_refinement_w_smooth_hand_rot,
                w_smooth_hand_articulation  = simple_refinement_w_smooth_hand_articulation,
                w_smooth_hand_trans         = simple_refinement_w_smooth_hand_trans,
                w_smooth_hand_object_relative_rot = simple_refinement_w_smooth_hand_object_relative_rot,
                w_smooth_hand_object_relative_trans = simple_refinement_w_smooth_hand_object_relative_trans,
                w_smooth_camera_rot         = simple_refinement_w_smooth_camera_rot,
                w_smooth_camera_trans       = simple_refinement_w_smooth_camera_trans,
                w_mask                      = simple_refinement_w_mask,
                w_relative_depth            = simple_refinement_w_relative_depth,
                w_perceptual                = simple_refinement_w_perceptual,
                vgg_weights_path            = simple_refinement_vgg_weights_path,
                perceptual_resize           = simple_refinement_perceptual_resize,
                w_hand_object_penetration   = simple_refinement_w_hand_object_penetration,
                hand_object_penetration_margin = simple_refinement_hand_object_penetration_margin,
                object_sdf_resolution       = simple_refinement_object_sdf_resolution,
                hand_object_penetration_max_verts = simple_refinement_hand_object_penetration_max_verts,
                mask_background             = simple_refinement_mask_background,
                init_opacity_obj            = init_opacity_obj_eff,
                init_opacity_hand           = init_opacity_hand_eff,
                init_opacity_bg             = init_opacity_bg_eff,
                init_gaussian_scale_factor  = init_gaussian_scale_factor,
                render_every                = simple_refinement_render_every,
                progress_dir                = refined_simple_progress_dir,
                debug_frame_idx             = (
                    simple_refinement_debug_frame
                    if simple_refinement_debug_frame is not None
                    else reference_frame
                ),
                bg_ref_frame                = reference_frame,
                bg_init_stride              = bg_init_stride,
                bg_voxel_size               = bg_voxel_size,
                bg_max_points               = bg_max_points,
                valid_mask_threshold        = valid_mask_threshold,
                valid_mask_erode_iters      = valid_mask_erode_iters,
                dev                         = dev,
            )

        # Render the refined simple-stage MANO/object poses through the same
        # 2x2 aligned overlay used by hamer_aligned_filled_overlay.mp4. The
        # simple optimizer folds learned object scale into the per-frame pose
        # JSONs before saving, so no extra object_scale multiplier is needed.
        if not _step(f"Render refined-simple {refine_source} overlay",
                     os.path.exists(refined_simple_hand_overlay)):
            if not _has_files(refined_simple_poses_dir):
                raise FileNotFoundError(
                    f"No refined simple object poses found in {refined_simple_poses_dir}"
                )
            if not _has_files(refined_simple_hand_dir):
                raise FileNotFoundError(
                    f"No refined simple hand tracks found in {refined_simple_hand_dir}"
                )
            run_render_hands_aligned_video(
                frames_dir         = frames_dir,
                aligned_dir        = refined_simple_hand_dir,
                mano_assets_root   = refine_mano_assets,
                output_path        = refined_simple_hand_overlay,
                object_mesh_path   = object_mesh_for_pipeline,
                object_poses_dir   = refined_simple_poses_dir,
                dev                = dev,
            )

    # Final: packaged result/ folder (mesh + textures + result.npz). The
    # object branch is required because the bundle is centered on the final
    # object mesh and object-to-camera trajectory; hand tracks are optional.
    result_dir = f"{output_dir}/result"
    result_npz = f"{result_dir}/result.npz"
    result_mesh = f"{result_dir}/mesh.obj"
    if object_mesh_for_pipeline is not None:
        final_object_poses_dir = _first_populated_dir([
            refined_simple_poses_dir,
            refined_poses_dir,
            poses_smooth_dir,
            poses_dir,
        ])

        primary_hand_roots = []
        if use_hamer_primary:
            primary_hand_roots.extend([
                hamer_aligned_filled_dir,
                hamer_aligned_dir,
                hamer_dir,
            ])
            primary_hand_roots.extend([
                wilor_aligned_filled_dir,
                wilor_aligned_dir,
                wilor_tracks_dir,
            ])
        else:
            primary_hand_roots.extend([
                wilor_aligned_filled_dir,
                wilor_aligned_dir,
                wilor_tracks_dir,
            ])
            if run_hamer_outputs:
                primary_hand_roots.extend([
                    hamer_aligned_filled_dir,
                    hamer_aligned_dir,
                    hamer_dir,
                ])
        final_hand_root = _first_populated_dir([
            refined_simple_hand_dir,
            refined_hand_dir,
            *primary_hand_roots,
        ])
        final_left_hand_dir, final_right_hand_dir = _result_hand_dirs(
            final_hand_root,
            hand_tracks,
        )

        final_camera_dir = None
        final_camera_convention = "camera_to_world"
        if _has_files(refined_simple_camera_dir):
            final_camera_dir = refined_simple_camera_dir
            final_camera_convention = "world_to_camera"
        elif run_slam and _has_files(slam_poses_dir):
            final_camera_dir = slam_poses_dir

        final_object_scale = 1.0
        if (final_object_poses_dir == refined_poses_dir
                and not using_provided_object_mesh):
            final_object_scale = _load_scale_json(refined_object_scale_json, 1.0)

        if final_object_poses_dir is None:
            print("  [skip] Package result/ (no object poses found)")
        elif not _step("Package result/", os.path.exists(result_npz)
                       and os.path.exists(result_mesh)):
            write_result_bundle(
                result_dir=result_dir,
                frames_dir=frames_dir,
                intrinsics_path=intrinsics_stable,
                mesh_path=object_mesh_for_pipeline,
                object_poses_dir=final_object_poses_dir,
                object_scale=final_object_scale,
                camera_to_world_dir=final_camera_dir,
                camera_pose_convention=final_camera_convention,
                left_hand_dir=final_left_hand_dir,
                right_hand_dir=final_right_hand_dir,
                source_manifest={
                    "pipeline": "ego_wilor",
                    "object_prompt": object_prompt,
                    "object_track_id": object_track_id,
                    "hand_pose_source": hand_pose_source,
                    "sam2_hand_prompt_source": sam2_hand_prompt_source,
                    "final_hand_root": final_hand_root,
                    "run_geocalib": run_geocalib_eff,
                    "geocalib_calibration_path": geocalib_calib if run_geocalib_eff else None,
                    "gravity_align": gravity_align,
                    "gravity_align_target": gravity_align_target,
                },
            )

        if gravity_align and final_object_poses_dir is not None:
            if not _step("Gravity-align result bundle",
                         result_bundle_has_gravity_alignment(result_dir, target=gravity_align_target)):
                gravity_align_result_bundle(
                    result_dir=result_dir,
                    geocalib_calibration_path=geocalib_calib,
                    target=gravity_align_target,
                )
    elif gravity_align:
        print("  [skip] Gravity-align result bundle (no object result bundle was produced)")

    print(f"\n{'='*60}")
    print(f"  Done.")
    if undistort:
        print(f"  anycalib        : {anycalib_dir}/")
    if run_geocalib_eff:
        print(f"  geocalib        : {geocalib_dir}/")
        print(f"  geocalib_gravity: {geocalib_gravity}")
    print(f"  wilor_raw       : {wilor_raw_dir}/")
    print(f"  hand_detections : {hand_detections}")
    print(f"  hand_tracks     : {hand_tracks}")
    print(f"  masks           : {masks_dir}/")
    if sam2_hand_prompt_source == "wilor_mask":
        print(f"  sam2_hand_prompt_masks: {sam2_hand_prompt_masks_dir}/")
    if refine_masks_with_silhouette:
        print(f"  wilor_filled    : {wilor_filled_dir}/  (real + interpolated)")
        print(f"  masks_refined   : {masks_refined_dir}/  (hand tracks only)")
        print(f"  masks_refined_overlay : {masks_refined_overlay}")
    print(f"  prompts_overlay : {prompts_overlay}")
    print(f"  masks_overlay   : {masks_overlay}")
    if not use_hamer_primary:
        print(f"  wilor (raw tracks)           : {wilor_tracks_dir}/")
        print(f"  wilor_overlay                : {wilor_overlay}")
        print(f"  wilor_aligned                : {wilor_aligned_dir}/  (real only)")
        print(f"  wilor_aligned/hand_scale.json: per-track multiplicative correction")
        print(f"  wilor_aligned_overlay        : {wilor_aligned_overlay}")
        print(f"  wilor_aligned_filled         : {wilor_aligned_filled_dir}/  (real + interpolated)")
        print(f"  wilor_aligned_filled_overlay : {wilor_aligned_filled_overlay}")
    if run_hamer_outputs:
        print(f"  hamer (per-track HaMeR)      : {hamer_dir}/")
        print(f"  hamer_overlay                : {hamer_overlay}")
        print(f"  hamer_aligned                : {hamer_aligned_dir}/")
        print(f"  hamer_aligned_overlay        : {hamer_aligned_overlay}")
        print(f"  hamer_aligned_filled         : {hamer_aligned_filled_dir}/")
        print(f"  hamer_aligned_filled_overlay : {hamer_aligned_filled_overlay}")
    if object_mesh_for_pipeline is not None:
        print(f"  object_mesh                  : {object_mesh_for_pipeline}")
        print(f"  result_bundle                : {result_dir}/")
        if gravity_align:
            print(f"  result_gravity_target        : {gravity_align_target}")
    if run_slam:
        print(f"  slam_poses                   : {slam_poses_dir}/")
        print(f"  slam_trajectory              : {slam_trajectory}")
    if run_refinement:
        print(f"  poses_refined                : {refined_poses_dir}/")
        print(f"  {refine_source}_refined                : {refined_hand_dir}/")
        print(f"  refined_overlay              : {refined_overlay}")
        print(f"  {refine_source}_refined_overlay        : {refined_hand_overlay}")
    if run_refinement_simple:
        print(f"  poses_refined_simple         : {refined_simple_poses_dir}/")
        print(f"  {refine_source}_refined_simple         : {refined_simple_hand_dir}/")
        print(f"  camera_refined_simple        : {refined_simple_camera_dir}/")
        print(f"  refined_simple_overlay       : {refined_simple_overlay}")
        print(f"  {refine_source}_refined_simple_overlay : {refined_simple_hand_overlay}")
        if simple_refinement_render_every > 0:
            print(f"  refined_simple_progress      : {refined_simple_progress_dir}/")
            print(f"  refined_simple_progress_mp4  : {refined_simple_progress_video}")
    print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_path",    required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--sam2_weights",  default="data/weights/sam2")
    p.add_argument("--wilor_weights", default="data/weights/wilor")
    p.add_argument("--reference_frame", type=int, default=0)
    p.add_argument("--sam2_bbox_pad", type=float, default=0.2,
                   help="Pad each WiLoR hand-detection bbox by this ratio "
                        "before passing to SAM2. 0 = use raw WiLoR bboxes.")
    p.add_argument("--no_sam2_square_wilor_bboxes",
                   dest="sam2_square_wilor_bboxes",
                   action="store_false",
                   default=True,
                   help="Do not square WiLoR hand-detection boxes before "
                        "passing them to SAM2. Object boxes are unaffected.")
    p.add_argument("--sam2_hand_prompt_source", default="box",
                   choices=("box", "wilor_mask"),
                   help="How to seed SAM2 hand tracks on the reference frame: "
                        "'box' uses WiLoR bboxes; 'wilor_mask' renders the "
                        "WiLoR MANO hand silhouette and uses it as a SAM2 "
                        "mask prompt.")
    p.add_argument("--undistort", action="store_true",
                   help="Run AnyCalib first to estimate intrinsics + distortion "
                        "and rebind video_path to the undistorted MP4.")
    p.add_argument("--anycalib_weights", default="data/weights/anycalib")
    p.add_argument("--run_geocalib", action="store_true",
                   help="Run GeoCalib on the current input video after optional "
                        "undistortion and write geocalib/{intrinsics,distortion,gravity,calibration}.json.")
    p.add_argument("--geocalib_weights_path", default="data/weights/geocalib",
                   help="Torch hub cache directory or checkpoint path for GeoCalib.")
    p.add_argument("--geocalib_weights", default="pinhole",
                   choices=("pinhole", "distorted"),
                   help="GeoCalib release weights to load when --geocalib_weights_path is a cache dir.")
    p.add_argument("--geocalib_camera_model", default="pinhole",
                   choices=("pinhole", "simple_radial", "radial", "simple_divisional"),
                   help="GeoCalib camera model. Non-pinhole models should use --geocalib_weights distorted.")
    p.add_argument("--geocalib_num_samples", type=int, default=8,
                   help="Number of uniformly sampled video frames for GeoCalib aggregation.")
    p.add_argument("--use_geocalib_intrinsics", action="store_true",
                   help="Pass GeoCalib intrinsics to MoGe as input intrinsics. "
                        "Also enables the GeoCalib step.")
    p.add_argument("--gravity_align", action="store_true",
                   help="Use GeoCalib gravity to rotate the packaged result/ "
                        "world transforms. Also enables the GeoCalib step.")
    p.add_argument("--gravity_align_target", default="z_up",
                   choices=("z_up", "z_down", "y_up", "y_down", "x_up", "x_down"),
                   help="World axis convention after --gravity_align. For example, "
                        "z_up maps gravity to [0, 0, -1].")
    p.add_argument("--moge_weights",     default="data/weights/moge")
    p.add_argument("--min_iou", type=float, default=0.1,
                   help="Minimum silhouette-vs-mask IoU for a wilor detection "
                        "to be assigned to a SAM2 track in a given frame.")
    p.add_argument("--mask_min_pixels", type=int, default=256,
                   help="During depth alignment, frames whose intersection "
                        "(rendered-MANO ∧ SAM2 hand mask) is below this "
                        "pixel count keep cam_t unchanged (dz=0).")
    p.add_argument("--interp_betas", default="fixed", choices=("fixed", "interp"),
                   help="MANO shape (betas) policy during interpolation: "
                        "'fixed' = median per track applied to every frame "
                        "(real + interpolated); 'interp' = per-frame linear.")
    p.add_argument("--interp_max_gap_frames", type=int, default=15,
                   help="Skip interpolating when bracketing real detections "
                        "are more than this many frames apart.")
    p.add_argument("--object_prompt", default=None,
                   help="Grounding-DINO text prompt for the held object "
                        "(e.g. 'blue cup'). When set, runs the object "
                        "branch: DINO → SAM2 → mesh source → FoundationPose.")
    p.add_argument("--object_mesh_path", default=None,
                   help="Existing metric OBJ to use for the object branch. "
                        "Skips SAM3D mesh generation and FoundationPose scale "
                        "estimation, then tracks this mesh directly.")
    p.add_argument("--grounding_dino_weights",  default="data/weights/grounding_dino")
    p.add_argument("--sam3d_weights",           default="data/weights/sam3d")
    p.add_argument("--foundation_pose_weights", default="data/weights/foundation_pose")
    p.add_argument("--reregister_iou_thresh", type=float, default=0.3,
                   help="FoundationPose re-register-from-scratch IoU threshold "
                        "(0 to disable).")
    p.add_argument("--run_slam", action="store_true",
                   help="Run DROID-SLAM after MoGe (scale-aligned to MoGe "
                        "depth). Per-frame Transform3d JSONs go to "
                        "<output_dir>/slam_poses/. When combined with "
                        "--run_refinement, they seed gsplat's background "
                        "pose field (critical for moving cameras).")
    p.add_argument("--droid_slam_weights", default="data/weights/droid_slam",
                   help="DROID-SLAM weights dir (used only with --run_slam).")
    p.add_argument("--bg_init_stride", type=int, default=10,
                   help="Stride for multi-frame BG point-cloud init at gsplat "
                        "refinement. Only effective with --run_slam (needs "
                        "per-frame poses to compose unprojections). Set to 1 "
                        "to force the old single-frame init even with SLAM.")
    p.add_argument("--bg_voxel_size", type=float, default=0.005,
                   help="Voxel size (m) for dedup of the fused BG point cloud "
                        "before random subsample. 0 disables voxel dedup.")
    p.add_argument("--bg_max_points", type=int, default=100000,
                   help="Cap on the BG Gaussian count after voxel dedup + "
                        "random subsample.")
    p.add_argument("--w_scale_aniso_bg", type=float, default=0.1,
                   help="SuGaR scale-anisotropy loss weight on the bg "
                        "Gaussians (0 disables). Pushes each Gaussian toward "
                        "a thin-disk shape; local, no neighbor coupling.")
    p.add_argument("--w_density_bg", type=float, default=0.0,
                   help="SuGaR density regularizer weight on the bg Gaussian "
                        "field (0 disables). Pulls the field toward a "
                        "surface-like (thin-shell) density profile in 3D. "
                        "Does NOT use depth images.")
    p.add_argument("--n_density_neighbors", type=int, default=8,
                   help="K for the density regularizer's nearest-neighbour "
                        "mixture sum.")
    p.add_argument("--density_subsample_frac_bg", type=float, default=0.2,
                   help="Fraction of bg Gaussians sampled as probe anchors "
                        "each step (cost/quality tradeoff).")
    p.add_argument("--w_sdf_density_bg", type=float, default=0.5,
                   help="SuGaR SDF loss weight on the bg Gaussians (0 "
                        "disables). Internally combines (a) |f̂−f| match "
                        "between depth-implied and Gaussian-mixture SDF "
                        "(Eq. 7-8) and (b) a samples-on-surface |f̂|/σ pull "
                        "(weight 0.2 from the paper) that moves probe "
                        "points — and their anchor Gaussians — onto the "
                        "depth surface. Uses the per-frame depth image.")
    p.add_argument("--w_normal_consistency_bg", type=float, default=0.1,
                   help="SuGaR paper-faithful normal-consistency loss weight "
                        "on the bg Gaussians (Eq. 10; 0 disables). Aligns the "
                        "density gradient direction with each Gaussian's "
                        "shortest axis (the surface normal of its disk).")
    p.add_argument("--n_sdf_samples_bg", type=int, default=20000,
                   help="Number of probe samples per step for the SuGaR SDF + "
                        "normal losses.")
    p.add_argument("--w_photometric_ssim", type=float, default=0.2,
                   help="SSIM photometric loss weight (1 - SSIM, 11x11 "
                        "Gaussian window). Captures local texture / edge "
                        "structure that pixel-wise L1 misses. 3DGS uses "
                        "~0.2 with L1=1.0.")
    p.add_argument("--w_depth_ssim", type=float, default=0.0,
                   help="SSIM depth loss weight (1 - SSIM on log-depth, "
                        "percentile-normalized). Off by default since pass 1 "
                        "leaves the L1 depth weight at 0.")
    p.add_argument("--valid_mask_threshold", type=float, default=0.12,
                   help="Max-brightness threshold (in [0,1] image scale) for "
                        "the static valid-pixel mask derived from the input "
                        "video. Pixels whose max brightness across all frames "
                        "is below this are treated as fixed dead/black "
                        "regions (fisheye crop, vignette) and excluded from "
                        "photometric / depth / SuGaR supervision. 0 disables.")
    p.add_argument("--valid_mask_erode_iters", type=int, default=3,
                   help="3x3 erosion passes on the valid-pixel mask, to peel "
                        "back the soft boundary transition.")
    p.add_argument("--n_sdf_neighbors_bg", type=int, default=16,
                   help="K for the K-nearest-Gaussian density mixture in the "
                        "SuGaR SDF + normal losses.")
    p.add_argument("--refine_masks_with_silhouette", action="store_true",
                   help="After wilor IoU matching, refine each hand's SAM2 mask "
                        "by intersecting it with the dilated rendered MANO "
                        "silhouette from wilor. Trims forearm bleed. "
                        "Object mask is left untouched.")
    p.add_argument("--mask_dilation_pixels", type=int, default=20,
                   help="Pixel radius of the dilation kernel applied to the "
                        "rendered MANO silhouette before intersecting with the "
                        "SAM2 mask (only used with --refine_masks_with_silhouette).")
    p.add_argument("--hand_pose_source", default="wilor", choices=("wilor", "hamer"),
                   help="Primary hand pose source after SAM2 propagation. "
                        "'wilor' keeps the original WiLoR detection-to-mask "
                        "matching path; 'hamer' uses WiLoR only to seed SAM2, "
                        "then runs HaMeR from each SAM2 mask bbox.")
    p.add_argument("--run_hamer_pass", action="store_true",
                   help="Also run HaMeR from the SAM2 hand masks and write "
                        "hamer*/ outputs. With --hand_pose_source hamer this "
                        "is implied and HaMeR becomes the primary hand source.")
    p.add_argument("--hamer_weights", default="data/weights/hamer",
                   help="HaMeR weights dir (used with --run_hamer_pass or "
                        "--hand_pose_source hamer).")
    p.add_argument("--hamer_bbox_expansion", type=float, default=1.7,
                   help="Bbox expansion for HaMeR's mask-driven crop.")
    p.add_argument("--hamer_mask_min_pixels", type=int, default=None,
                   help="Min SAM2-mask area for HaMeR alignment / regression. "
                        "Defaults to 1 with --hand_pose_source hamer, or 256 "
                        "when HaMeR is only an optional sidecar pass.")
    p.add_argument("--run_refinement", action="store_true",
                   help="After alignment, jointly refine hand+object poses via "
                        "Gaussian splatting. Requires --object_prompt.")
    p.add_argument("--run_refinement_simple", action="store_true",
                   help="Run the simplified 2DGS refinement path: face-anchored "
                        "object/hand Gaussians, depth-initialized background, "
                        "L1 photometric loss, root-pose/camera-pose updates, "
                        "and optional finger/shape/scale updates. Requires --object_prompt.")
    p.add_argument("--simple_refinement_epochs", type=int, default=15)
    p.add_argument("--simple_refinement_batch_size", type=int, default=8)
    p.add_argument("--simple_refinement_train_resolution_scale", type=float, default=0.5)
    p.add_argument("--simple_refinement_lr_gaussians", type=float, default=1e-4)
    p.add_argument("--simple_refinement_lr_object_pose", type=float, default=1e-4)
    p.add_argument("--simple_refinement_lr_object_scale", type=float, default=0.0,
                   help="LR for simple-refinement global object scale. 0 disables.")
    p.add_argument("--simple_refinement_lr_hand_pose", type=float, default=1e-4,
                   help="LR for simple-refinement hand root global_orient and cam_t. 0 disables root pose refinement.")
    p.add_argument("--simple_refinement_lr_hand_articulation", type=float, default=1e-4,
                   help="LR for simple-refinement MANO finger articulation hand_pose. 0 disables.")
    p.add_argument("--simple_refinement_lr_hand_shape", type=float, default=1e-4,
                   help="LR for simple-refinement shared MANO betas/shape per hand track. 0 disables.")
    p.add_argument("--simple_refinement_lr_hand_scale", type=float, default=1e-4,
                   help="LR for simple-refinement shared per-track hand_scale. 0 disables.")
    p.add_argument("--simple_refinement_lr_camera_pose", type=float, default=1e-4)
    p.add_argument("--simple_refinement_lr_schedule", choices=["constant", "cosine"], default="constant",
                   help="Learning-rate schedule for simple refinement optimizer groups.")
    p.add_argument("--simple_refinement_lr_cosine_min_factor", type=float, default=0.0,
                   help="Final LR multiplier for simple-refinement cosine LR. 0 decays to zero.")
    p.add_argument("--simple_refinement_w_smooth_object_rot", type=float, default=15.0)
    p.add_argument("--simple_refinement_w_smooth_object_trans", type=float, default=15.0)
    p.add_argument("--simple_refinement_w_smooth_hand_rot", type=float, default=15.0)
    p.add_argument("--simple_refinement_w_smooth_hand_articulation", type=float, default=1.0,
                   help="Temporal rotation smoothness for simple-refinement MANO finger hand_pose.")
    p.add_argument("--simple_refinement_w_smooth_hand_trans", type=float, default=15.0)
    p.add_argument("--simple_refinement_w_smooth_hand_object_relative_rot", type=float, default=0.0,
                   help="Temporal smoothness for simple-refinement hand root rotation relative "
                        "to object pose. 0 disables.")
    p.add_argument("--simple_refinement_w_smooth_hand_object_relative_trans", type=float, default=0.0,
                   help="Temporal smoothness for simple-refinement hand root translation in the "
                        "object frame. 0 disables.")
    p.add_argument("--simple_refinement_w_smooth_camera_rot", type=float, default=15.0)
    p.add_argument("--simple_refinement_w_smooth_camera_trans", type=float, default=15.0)
    p.add_argument("--simple_refinement_w_mask", type=float, default=0.0,
                   help="Weight for simple-refinement L1 segmentation mask loss. 0 disables.")
    p.add_argument("--simple_refinement_w_relative_depth", type=float, default=0.0,
                   help="Weight for simple-refinement log-depth-gradient loss against MoGe depth. 0 disables.")
    p.add_argument("--simple_refinement_w_perceptual", type=float, default=0.0,
                   help="Weight for simple-refinement VGG16 perceptual feature L1 loss. 0 disables.")
    p.add_argument("--simple_refinement_vgg_weights_path", default=None,
                   help="Optional local VGG16 state_dict path for simple refinement. If missing, downloaded there.")
    p.add_argument("--simple_refinement_perceptual_resize", type=int, default=224,
                   help="Square resize for simple-refinement VGG perceptual loss. <=0 uses training resolution.")
    p.add_argument("--simple_refinement_w_hand_object_penetration", type=float, default=0.0,
                   help="Weight for simple-refinement 3D hand/object SDF penetration loss. 0 disables.")
    p.add_argument("--simple_refinement_hand_object_penetration_margin", type=float, default=0.003,
                   help="Signed-distance margin in meters/object units for simple-refinement hand/object separation.")
    p.add_argument("--simple_refinement_object_sdf_resolution", type=int, default=96,
                   help="Resolution per axis for simple-refinement object SDF grid.")
    p.add_argument("--simple_refinement_hand_object_penetration_max_verts", type=int, default=0,
                   help="Max MANO verts per hand for simple-refinement penetration loss. 0 uses all vertices.")
    p.add_argument("--simple_refinement_mask_background", action="store_true",
                   help="For simple refinement, black out target background and omit background Gaussians/camera refinement.")
    p.add_argument("--simple_refinement_render_every", type=int, default=25,
                   help="Write simple-refinement progress panels every N optimizer steps. 0 disables.")
    p.add_argument("--simple_refinement_debug_frame", type=int, default=None,
                   help="Frame index to visualize for simple-refinement progress. Defaults to --reference_frame.")
    p.add_argument("--learn_hand_scale", action="store_true",
                   help="Optimize the per-track hand_scale during gsplat "
                        "refinement (init from align_hands' median). Off by "
                        "default; the init is already a strong estimate and "
                        "`betas` covers the same DoF.")
    p.add_argument("--lr_hand_scale", type=float, default=1e-3,
                   help="LR for hand_scale when --learn_hand_scale is set.")
    p.add_argument("--w_hand_scale_prior", type=float, default=10.0,
                   help="Tight prior pulling learned hand_scale back to the "
                        "align_hands estimate.")
    p.add_argument("--refinement_render_every", type=int, default=25,
                   help="Dump an overlay PNG of a reference frame every N "
                        "optimizer steps during refinement (0 disables).")
    p.add_argument("--w_smooth_obj_rot", type=float, default=1.0)
    p.add_argument("--w_smooth_obj_trans", type=float, default=1.0)
    p.add_argument("--w_smooth_hand_rot", type=float, default=1.0)
    p.add_argument("--w_smooth_hand_finger", type=float, default=1.0)
    p.add_argument("--w_smooth_hand_trans", type=float, default=1.0)
    p.add_argument("--w_smooth_bg_rot", type=float, default=1.0)
    p.add_argument("--w_smooth_bg_trans", type=float, default=1.0)
    p.add_argument("--smooth_obj_in_world", action="store_true",
                   help="Measure full-refinement object smoothness in world frame instead of camera frame.")
    p.add_argument("--smooth_hand_in_world", action="store_true",
                   help="Measure full-refinement hand wrist smoothness in world frame instead of camera frame.")
    p.add_argument("--rotation_search_smoothness_weight", type=float, default=1.0,
                   help="Causal smoothness weight used by optional rotation-search candidate scoring.")
    p.add_argument("--init_opacity", type=float, default=None,
                   help="Shared initial opacity for object, hand, background, and wrist Gaussians. "
                        "Per-set --init_opacity_* flags override this.")
    p.add_argument("--init_opacity_obj", type=float, default=0.1)
    p.add_argument("--init_opacity_hand", type=float, default=0.1)
    p.add_argument("--init_opacity_bg", type=float, default=0.1)
    p.add_argument("--init_opacity_wrist", type=float, default=None)
    p.add_argument("--init_gaussian_scale_factor", type=float, default=2.0,
                   help="Multiplier for all initial Gaussian scales/radii in refinement.")
    p.add_argument("--refinement_resume", action="store_true",
                   help="Resume gsplat refinement from "
                        "<output_dir>/refine_checkpoint.pt if present.")
    p.add_argument("--refinement_object_pose_only", action="store_true",
                   help="Freeze Gaussians + hand pose + bg pose; only the "
                        "object pose updates. Usually combined with "
                        "--refinement_resume.")
    p.add_argument("--refinement_random_init_obj_pose", action="store_true",
                   help="Randomize per-frame object pose before training. "
                        "Tests whether the optimizer can recover the pose "
                        "from scratch.")
    p.add_argument("--refinement_ignore_optimizer_state", action="store_true",
                   help="When resuming, skip loading optimizer/LR-scheduler "
                        "state — Adam moments reinit lazily.")
    p.add_argument("--refinement_second_pass", action="store_true",
                   help="After the first refinement, run a second polishing "
                        "pass (resumes from pass-1's checkpoint, lower LRs, "
                        "larger batch, no warmup).")
    p.add_argument("--refinement_second_pass_lr_scale", type=float, default=0.3,
                   help="Multiplier applied to every LR in pass 2.")
    p.add_argument("--refinement_second_pass_epochs", type=int, default=None,
                   help="Number of pass-2 epochs. Defaults to half of pass-1.")
    p.add_argument("--refinement_second_pass_batch_size", type=int, default=None,
                   help="Batch size for pass 2. Defaults to 4x pass-1.")
    p.add_argument("--dev", action="store_true")
    return p.parse_args()


def run_from_args(args: argparse.Namespace) -> None:
    run_ego_wilor(
        video_path              = args.video_path,
        output_dir              = args.output_dir,
        sam2_weights            = args.sam2_weights,
        wilor_weights           = args.wilor_weights,
        reference_frame         = args.reference_frame,
        sam2_bbox_pad           = args.sam2_bbox_pad,
        sam2_square_wilor_bboxes = args.sam2_square_wilor_bboxes,
        sam2_hand_prompt_source = args.sam2_hand_prompt_source,
        undistort               = args.undistort,
        anycalib_weights        = args.anycalib_weights,
        run_geocalib            = args.run_geocalib,
        geocalib_weights_path   = args.geocalib_weights_path,
        geocalib_weights        = args.geocalib_weights,
        geocalib_camera_model   = args.geocalib_camera_model,
        geocalib_num_samples    = args.geocalib_num_samples,
        use_geocalib_intrinsics = args.use_geocalib_intrinsics,
        gravity_align           = args.gravity_align,
        gravity_align_target    = args.gravity_align_target,
        moge_weights            = args.moge_weights,
        min_iou                 = args.min_iou,
        mask_min_pixels         = args.mask_min_pixels,
        interp_betas            = args.interp_betas,
        interp_max_gap_frames   = args.interp_max_gap_frames,
        object_prompt           = args.object_prompt,
        object_mesh_path        = args.object_mesh_path,
        grounding_dino_weights  = args.grounding_dino_weights,
        sam3d_weights           = args.sam3d_weights,
        foundation_pose_weights = args.foundation_pose_weights,
        reregister_iou_thresh   = args.reregister_iou_thresh,
        run_slam                = args.run_slam,
        droid_slam_weights      = args.droid_slam_weights,
        bg_init_stride          = args.bg_init_stride,
        bg_voxel_size           = args.bg_voxel_size,
        bg_max_points           = args.bg_max_points,
        w_scale_aniso_bg        = args.w_scale_aniso_bg,
        w_density_bg            = args.w_density_bg,
        n_density_neighbors     = args.n_density_neighbors,
        density_subsample_frac_bg = args.density_subsample_frac_bg,
        w_sdf_density_bg          = args.w_sdf_density_bg,
        w_normal_consistency_bg   = args.w_normal_consistency_bg,
        n_sdf_samples_bg          = args.n_sdf_samples_bg,
        n_sdf_neighbors_bg        = args.n_sdf_neighbors_bg,
        w_photometric_ssim        = args.w_photometric_ssim,
        w_depth_ssim              = args.w_depth_ssim,
        valid_mask_threshold      = args.valid_mask_threshold,
        valid_mask_erode_iters    = args.valid_mask_erode_iters,
        refine_masks_with_silhouette      = args.refine_masks_with_silhouette,
        mask_dilation_pixels              = args.mask_dilation_pixels,
        hand_pose_source                  = args.hand_pose_source,
        run_hamer_pass                    = args.run_hamer_pass,
        hamer_weights                     = args.hamer_weights,
        hamer_bbox_expansion              = args.hamer_bbox_expansion,
        hamer_mask_min_pixels             = args.hamer_mask_min_pixels,
        run_refinement                    = args.run_refinement,
        run_refinement_simple             = args.run_refinement_simple,
        simple_refinement_epochs          = args.simple_refinement_epochs,
        simple_refinement_batch_size      = args.simple_refinement_batch_size,
        simple_refinement_train_resolution_scale = args.simple_refinement_train_resolution_scale,
        simple_refinement_lr_gaussians    = args.simple_refinement_lr_gaussians,
        simple_refinement_lr_object_pose  = args.simple_refinement_lr_object_pose,
        simple_refinement_lr_object_scale = args.simple_refinement_lr_object_scale,
        simple_refinement_lr_hand_pose    = args.simple_refinement_lr_hand_pose,
        simple_refinement_lr_hand_articulation = args.simple_refinement_lr_hand_articulation,
        simple_refinement_lr_hand_shape   = args.simple_refinement_lr_hand_shape,
        simple_refinement_lr_hand_scale   = args.simple_refinement_lr_hand_scale,
        simple_refinement_lr_camera_pose  = args.simple_refinement_lr_camera_pose,
        simple_refinement_lr_schedule     = args.simple_refinement_lr_schedule,
        simple_refinement_lr_cosine_min_factor = args.simple_refinement_lr_cosine_min_factor,
        simple_refinement_w_smooth_object_rot = args.simple_refinement_w_smooth_object_rot,
        simple_refinement_w_smooth_object_trans = args.simple_refinement_w_smooth_object_trans,
        simple_refinement_w_smooth_hand_rot = args.simple_refinement_w_smooth_hand_rot,
        simple_refinement_w_smooth_hand_articulation = args.simple_refinement_w_smooth_hand_articulation,
        simple_refinement_w_smooth_hand_trans = args.simple_refinement_w_smooth_hand_trans,
        simple_refinement_w_smooth_hand_object_relative_rot = args.simple_refinement_w_smooth_hand_object_relative_rot,
        simple_refinement_w_smooth_hand_object_relative_trans = args.simple_refinement_w_smooth_hand_object_relative_trans,
        simple_refinement_w_smooth_camera_rot = args.simple_refinement_w_smooth_camera_rot,
        simple_refinement_w_smooth_camera_trans = args.simple_refinement_w_smooth_camera_trans,
        simple_refinement_w_mask          = args.simple_refinement_w_mask,
        simple_refinement_w_relative_depth = args.simple_refinement_w_relative_depth,
        simple_refinement_w_perceptual    = args.simple_refinement_w_perceptual,
        simple_refinement_vgg_weights_path = args.simple_refinement_vgg_weights_path,
        simple_refinement_perceptual_resize = args.simple_refinement_perceptual_resize,
        simple_refinement_w_hand_object_penetration = args.simple_refinement_w_hand_object_penetration,
        simple_refinement_hand_object_penetration_margin = args.simple_refinement_hand_object_penetration_margin,
        simple_refinement_object_sdf_resolution = args.simple_refinement_object_sdf_resolution,
        simple_refinement_hand_object_penetration_max_verts = args.simple_refinement_hand_object_penetration_max_verts,
        simple_refinement_mask_background = args.simple_refinement_mask_background,
        simple_refinement_render_every    = args.simple_refinement_render_every,
        simple_refinement_debug_frame     = args.simple_refinement_debug_frame,
        learn_hand_scale                  = args.learn_hand_scale,
        lr_hand_scale                     = args.lr_hand_scale,
        w_hand_scale_prior                = args.w_hand_scale_prior,
        refinement_render_every           = args.refinement_render_every,
        w_smooth_obj_rot                  = args.w_smooth_obj_rot,
        w_smooth_obj_trans                = args.w_smooth_obj_trans,
        w_smooth_hand_rot                 = args.w_smooth_hand_rot,
        w_smooth_hand_finger              = args.w_smooth_hand_finger,
        w_smooth_hand_trans               = args.w_smooth_hand_trans,
        w_smooth_bg_rot                   = args.w_smooth_bg_rot,
        w_smooth_bg_trans                 = args.w_smooth_bg_trans,
        smooth_obj_in_world               = args.smooth_obj_in_world,
        smooth_hand_in_world              = args.smooth_hand_in_world,
        rotation_search_smoothness_weight = args.rotation_search_smoothness_weight,
        init_opacity                      = args.init_opacity,
        init_opacity_obj                  = args.init_opacity_obj,
        init_opacity_hand                 = args.init_opacity_hand,
        init_opacity_bg                   = args.init_opacity_bg,
        init_opacity_wrist                = args.init_opacity_wrist,
        init_gaussian_scale_factor        = args.init_gaussian_scale_factor,
        refinement_resume                 = args.refinement_resume,
        refinement_object_pose_only       = args.refinement_object_pose_only,
        refinement_random_init_obj_pose   = args.refinement_random_init_obj_pose,
        refinement_ignore_optimizer_state = args.refinement_ignore_optimizer_state,
        refinement_second_pass            = args.refinement_second_pass,
        refinement_second_pass_lr_scale   = args.refinement_second_pass_lr_scale,
        refinement_second_pass_epochs     = args.refinement_second_pass_epochs,
        refinement_second_pass_batch_size = args.refinement_second_pass_batch_size,
        dev                     = args.dev,
    )


if __name__ == "__main__":
    run_from_args(parse_args())
