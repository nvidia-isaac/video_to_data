# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hand pipeline: detection → SAM2 tracking → HaMeR → depth alignment.

Steps:
  0.  AnyCalib undistortion           [optional, --undistort]
  1.  Extract frames
  2.  MoGe depth + intrinsics         [only when --run_hamer]
  2b. Stabilise intrinsics            [only when --run_hamer]
  3.  MediaPipe HandLandmarker        → hand_detections.json
  4.  Build SAM2 prompts               → sam2_prompts.json + hand_tracks.json
  5.  SAM2 mask propagation            → masks/{1,2,...}/*.png
  6.  Render prompts overlay PNG + mask overlay MP4
  7.  HaMeR per-track per-frame        [--run_hamer]
  8.  Render HaMeR mesh overlay        [--run_hamer]
  9.  Align HaMeR to depth             [--run_hamer]
  10. Render aligned HaMeR overlay     [--run_hamer]
  11. Joint hand+object GS refinement  [--run_refinement, requires --run_hamer
                                         and --object_prompt]
  12. Render refined HaMeR overlay     [--run_refinement] (mesh raster
                                         using refined poses; A/B vs step 10)

Output directory layout (with --undistort + --run_hamer):
  <output_dir>/
  ├── anycalib/{intrinsics,distortion}.json
  │   ├── undistorted.mp4
  │   └── undistorted_intrinsics.json
  ├── frames/
  ├── depth/                       # MoGe depth PNGs
  ├── intrinsics/                  # MoGe per-frame intrinsics JSONs
  ├── intrinsics_stable.json       # stabilised single-JSON intrinsics
  ├── hand_detections.json
  ├── sam2_prompts.json
  ├── hand_tracks.json
  ├── prompts_overlay.png
  ├── masks_overlay.mp4
  ├── masks/{1,2}/                 # SAM2 hand masks per track
  ├── hamer/{1,2}/*.json           # HaMeR per-frame MANO + virtual cam
  ├── hamer_overlay.mp4
  ├── hamer_aligned/{1,2}/*.json   # depth-aligned, real-intrinsics
  ├── hamer_aligned_overlay.mp4
  ├── poses_refined/*.json         # [--run_refinement]
  ├── hamer_refined/{2,3}/*.json   # [--run_refinement]
  ├── refined_overlay.mp4          # [--run_refinement] gsplat-rendered
  └── hamer_refined_overlay.mp4    # [--run_refinement] mesh raster

Usage:
    python modules/v2d_pipelines/run_hand_masks.py \\
        --video_path data/my_video.mp4 \\
        --output_dir data/outputs/my_video \\
        --sam2_weights data/weights/sam2 \\
        --undistort \\
        --run_hamer

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
    """Apply SAM3D rotation+scale (no translation) to mesh vertices.

    Mirrors the helper in run_v2d_ego_e2e.py — bakes R·s into vertices so
    FoundationPose sees a metric-scaled mesh.
    """
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

from v2d.anycalib.docker.run_video_to_calibration import run_video_to_calibration
from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.common.utils import extract_images
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.gsplat_refinement.docker.run_refine import run_refine
from v2d.hamer.docker.run_align_hands import run_align_hands
from v2d.hamer.docker.run_masks_to_hands import run_masks_to_hands
from v2d.hamer.docker.run_render_hands_aligned_video import run_render_hands_aligned_video
from v2d.hamer.docker.run_render_hands_video import run_render_hands_video
from v2d.mediapipe.docker.run_image_to_hand_bboxes import run_image_to_hand_bboxes
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_moge_depth
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh


# Distinct colors for up to ~10 tracks. Track id 1 → palette[0], etc.
_TRACK_COLORS = [
    (255,  60,  60),   # red
    ( 60, 160, 255),   # blue
    ( 60, 200,  60),   # green
    (255, 180,  40),   # orange
    (200,  80, 220),   # purple
    ( 60, 220, 220),   # cyan
    (255, 240,  60),   # yellow
    (240, 100, 180),   # pink
    (180, 220, 100),   # lime
    (140, 110,  80),   # brown
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
    """Save reference frame with bbox prompts drawn per track."""
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
    """Encode a video with per-track colored mask overlays on every frame."""
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
                m = (mask > 127).astype(np.float32)[..., None]   # (H, W, 1)
                color = np.array(_track_color(oid), dtype=np.float32)[None, None, :]
                frame = frame * (1 - alpha * m) + color * (alpha * m)
            out_img = Image.fromarray(frame.clip(0, 255).astype(np.uint8))
            # Legend: small color swatches + "id=N (L|R)" labels in the corner
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


def _plot_pose_grad_from_checkpoint(checkpoint_path: str, output_png: str) -> None:
    """Read the gsplat refinement checkpoint and plot per-frame pose-grad RMS.

    The checkpoint's ``pose_grad_state`` block holds Adam's ``exp_avg_sq``
    (EMA of squared gradients) for each pose parameter, labeled by name.
    Per-frame RMS magnitude is ``sqrt(sum(exp_avg_sq, dim=axes_after_T))``;
    high values indicate frames whose pose was *still being actively
    updated* at the end of training — usually "uncertain" / contested.

    Pure host-side debugging utility; safe to no-op if matplotlib or the
    checkpoint isn't present.
    """
    try:
        import matplotlib.pyplot as plt
        import torch
    except ImportError as e:
        print(f"  Skipping pose-grad plot: missing dependency ({e}).")
        return
    if not os.path.exists(checkpoint_path):
        print(f"  Skipping pose-grad plot: {checkpoint_path} not found.")
        return

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pg = ckpt.get("pose_grad_state")
    if pg is None:
        print(f"  Checkpoint has no pose_grad_state — refusing to plot.")
        return

    # Use actual on-disk frame indices for the x-axis when available, so
    # spikes can be matched directly to a frame in the source video. Falls
    # back to positional indices if the checkpoint didn't save them.
    frame_indices = ckpt.get("frame_indices")
    if frame_indices is not None:
        frame_indices = np.asarray(frame_indices, dtype=np.int64)
        xlabel = "frame index"
    else:
        xlabel = "frame index (positional)"

    def _rms(t):
        if t is None: return None
        t = t.detach().to(dtype=torch.float32)
        if t.dim() >= 2:
            t = t.reshape(t.shape[0], -1).sum(dim=-1)
        return t.sqrt().cpu().numpy()

    series: list[tuple[str, np.ndarray]] = []
    if (v := _rms(pg.get("obj_axis_angle")))  is not None: series.append(("object axis_angle", v))
    if (v := _rms(pg.get("obj_translation"))) is not None: series.append(("object translation", v))
    for hi, h in enumerate(pg.get("hands", [])):
        side = h.get("side", f"h{hi}")
        for name in ("global_orient", "hand_pose", "cam_t"):
            v = _rms(h.get(name))
            if v is not None:
                series.append((f"{side} hand {name}", v))

    if not series:
        print(f"  Pose-grad state is empty — nothing to plot.")
        return

    n_panels = len(series)
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(10, max(2, 1.6 * n_panels)),
        sharex=True, squeeze=False,
    )
    for ax, (label, v) in zip(axes[:, 0], series):
        # If frame_indices length doesn't match, fall back to positional.
        if frame_indices is not None and len(frame_indices) == len(v):
            xs = frame_indices
        else:
            xs = np.arange(len(v))
        ax.plot(xs, v, lw=0.9)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel(xlabel)
    fig.suptitle("Per-frame Adam exp_avg_sq RMS (high = uncertain pose)",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)) or ".", exist_ok=True)
    fig.savefig(output_png, dpi=120)
    plt.close(fig)
    print(f"  Wrote pose-grad plot → {output_png}")


def _step(label: str, done: bool) -> bool:
    if done:
        print(f"  [skip] {label}")
        return True
    print(f"  [run ] {label}")
    return False


def _pad_and_square_bbox(
    bbox: dict, pad_ratio: float, img_w: int, img_h: int,
) -> dict:
    """Center the bbox, expand to square at the longer side x (1+pad_ratio),
    then clamp to image bounds.

    SAM2 tracks better when its prompt bbox includes context around the
    target. MediaPipe's hand detections are tight on the visible hand
    silhouette — a short pad+square pass adds margin and gives SAM2 a
    consistent aspect ratio across all hand prompts. Clamping to image
    bounds may slightly reduce squareness near the edges; that's
    acceptable (SAM2 doesn't require strictly square prompts).
    """
    x0, y0, x1, y1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * (1.0 + pad_ratio)
    half = side / 2.0
    return {
        "x0": max(0.0,            cx - half),
        "y0": max(0.0,            cy - half),
        "x1": min(float(img_w),   cx + half),
        "y1": min(float(img_h),   cy + half),
    }


def run_hand_masks(
    video_path: str,
    output_dir: str,
    sam2_weights: str,
    reference_frame: int = 0,
    selfie: bool = False,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.1,
    pad_ratio: float = 0.15,
    sam2_bbox_pad: float = 0.2,
    undistort: bool = False,
    anycalib_weights: str = "data/weights/anycalib",
    moge_weights: str = "data/weights/moge",
    run_hamer: bool = False,
    hamer_weights: str | None = None,
    bbox_expansion: float = 1.7,
    mask_min_pixels: int = 256,
    object_prompt: str | None = None,
    grounding_dino_weights: str = "data/weights/grounding_dino",
    sam3d_weights: str = "data/weights/sam3d",
    foundation_pose_weights: str = "data/weights/foundation_pose",
    reregister_iou_thresh: float | None = 0.3,
    run_refinement: bool = False,
    refinement_epochs: int = 30,
    refinement_batch_size: int = 4,
    refinement_render_every: int = 25,
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
    os.makedirs(output_dir, exist_ok=True)

    frames_dir         = f"{output_dir}/frames"
    depth_dir          = f"{output_dir}/depth"
    intrinsics_dir     = f"{output_dir}/intrinsics"
    intrinsics_stable  = f"{output_dir}/intrinsics_stable.json"
    anycalib_dir       = f"{output_dir}/anycalib"
    anycalib_intr      = f"{anycalib_dir}/intrinsics.json"
    anycalib_dist      = f"{anycalib_dir}/distortion.json"
    undistorted_video  = f"{anycalib_dir}/undistorted.mp4"
    undistorted_intr   = f"{anycalib_dir}/undistorted_intrinsics.json"
    hand_detections    = f"{output_dir}/hand_detections.json"
    dino_detections    = f"{output_dir}/dino_detections.json"
    sam2_prompts       = f"{output_dir}/sam2_prompts.json"
    hand_tracks        = f"{output_dir}/hand_tracks.json"
    object_track       = f"{output_dir}/object_track.json"
    masks_dir          = f"{output_dir}/masks"
    prompts_overlay    = f"{output_dir}/prompts_overlay.png"
    masks_overlay      = f"{output_dir}/masks_overlay.mp4"
    # Object-branch outputs
    mesh_dir           = f"{output_dir}/mesh"
    mesh_path          = f"{mesh_dir}/textured_mesh.obj"
    mesh_transform     = f"{mesh_dir}/mesh_transform.json"
    mesh_intrinsics    = f"{mesh_dir}/mesh_intrinsics.json"
    mesh_pretransformed= f"{output_dir}/mesh_pretransformed.obj"
    mesh_scaled        = f"{output_dir}/mesh_scaled.obj"
    scale_path         = f"{output_dir}/scale.json"
    poses_dir          = f"{output_dir}/poses"
    poses_smooth_dir   = f"{output_dir}/poses_smoothed"
    ref_rgb            = f"{frames_dir}/{reference_frame:06d}.png"
    ref_depth          = f"{depth_dir}/{reference_frame:06d}.png"

    print(f"\n{'='*60}")
    print(f"  video           : {os.path.basename(video_path)}")
    print(f"  output          : {output_dir}")
    print(f"  reference_frame : {reference_frame}")
    print(f"  selfie          : {selfie}")
    print(f"  undistort       : {undistort}")
    print(f"  run_hamer       : {run_hamer}")
    print(f"{'='*60}\n")

    # 0. AnyCalib undistortion (optional) ------------------------------------
    if undistort:
        os.makedirs(anycalib_dir, exist_ok=True)
        if not _step("AnyCalib undistortion", os.path.exists(undistorted_video)):
            run_video_to_calibration(
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

    # 1. Extract frames -------------------------------------------------------
    if not _step("Extract frames", _has_files(frames_dir)):
        extract_images(video_path, frames_dir)

    # 2. MoGe depth + intrinsics + stabilise (only if HaMeR alignment will run)
    if run_hamer:
        moge_input_intrinsics = undistorted_intr if undistort else None
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

    # 2. Hand detection on reference frame ------------------------------------
    # Treat an existing-but-empty detections file as "redo" — otherwise a
    # cached failure (no hands found) would short-circuit every later run.
    cached_nonempty = (
        os.path.exists(hand_detections)
        and os.path.getsize(hand_detections) > len(b"[]\n")
    )
    if not _step("Hand detection (MediaPipe)", cached_nonempty):
        run_image_to_hand_bboxes(
            image_path               = ref_rgb,
            output_path              = hand_detections,
            max_num_hands            = max_num_hands,
            min_detection_confidence = min_detection_confidence,
            pad_ratio                = pad_ratio,
            selfie                   = selfie,
            dev                      = dev,
        )

    with open(hand_detections) as f:
        detections = json.load(f)
    if not detections:
        raise RuntimeError(
            f"Hand detector found no hands in {ref_rgb}.  Things to try:\n"
            f"  • lower --min_detection_confidence (current: {min_detection_confidence})\n"
            f"  • pick a different --reference_frame where hands are more visible\n"
            f"  • toggle --selfie if footage is mirrored\n"
            f"MediaPipe Hands is weak on egocentric views; a different detector "
            f"may be needed.  Delete {hand_detections} before retrying."
        )

    # Pad + square hand bboxes before they're consumed by SAM2 / overlays.
    # The on-disk hand_detections.json keeps the raw MediaPipe boxes; only
    # the in-memory copy used downstream is mutated.
    if sam2_bbox_pad > 0.0:
        ref_img = Image.open(ref_rgb)
        ref_w, ref_h = ref_img.size
        for det in detections:
            det["bbox"] = _pad_and_square_bbox(
                det["bbox"], sam2_bbox_pad, ref_w, ref_h,
            )

    # 2b. Object detection (optional) -----------------------------------------
    # When --object_prompt is set, run Grounding DINO on the reference frame
    # and reserve object_id=1 for the held object. Hands then take 2..N+1.
    object_track_id: int | None = None
    if object_prompt is not None:
        if not _step("Grounding DINO (frame 0)", os.path.exists(dino_detections)):
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

    # 3. Build SAM2 prompts ---------------------------------------------------
    # object_id allocation: object (if any) gets 1, hands get 2..N+1.
    # When no object, hands keep 1..N (back-compat with prior runs).
    if not (os.path.exists(sam2_prompts) and os.path.exists(hand_tracks)):
        prompts: list[Sam2Prompt] = []
        hand_meta: list[dict] = []
        if object_track_id is not None:
            prompts.append(Sam2Prompt(
                frame_index=reference_frame,
                object_id=object_track_id,
                box=object_box,
            ))
        hand_id_offset = 1 if object_track_id is None else 2
        for i, det in enumerate(detections):
            oid = i + hand_id_offset
            prompts.append(Sam2Prompt(
                frame_index=reference_frame,
                object_id=oid,
                box=BoundingBox.from_dict(det["bbox"]),
            ))
            hand_meta.append({
                "object_id": oid,
                "role":      "hand",
                "is_right":  bool(det["is_right"]),
                "score":     float(det["score"]),
            })
        Sam2Prompts(prompts=prompts).save(sam2_prompts)
        with open(hand_tracks, "w") as f:
            json.dump({
                "reference_frame": reference_frame,
                "tracks":          hand_meta,
            }, f, indent=2)
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

    # 4. SAM2 video propagation -----------------------------------------------
    # Already-propagated runs are detected by the presence of any mask subdir.
    sam2_done = any(
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

    # 5. Verification renders -------------------------------------------------
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

    # 6. Object branch: SAM3D mesh → FoundationPose tracking ----------------
    # Requires depth + intrinsics, so gated on run_hamer (which produces them).
    if object_prompt is not None and run_hamer:
        ref_obj_mask = f"{masks_dir}/{object_track_id}/{reference_frame:06d}.png"
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

        if not _step("FoundationPose tracking", _has_files(poses_dir)):
            run_video_to_poses(
                video_path             = video_path,
                depth_folder           = depth_dir,
                masks_folder           = f"{masks_dir}/{object_track_id}",
                camera_intrinsics_path = intrinsics_stable,
                mesh_path              = mesh_scaled,
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
                mesh_path            = mesh_scaled,
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

    # 7. Optional HaMeR per-frame reconstruction -----------------------------
    hamer_dir             = f"{output_dir}/hamer"
    hamer_overlay         = f"{output_dir}/hamer_overlay.mp4"
    hamer_aligned_dir     = f"{output_dir}/hamer_aligned"
    hamer_aligned_overlay = f"{output_dir}/hamer_aligned_overlay.mp4"
    if run_hamer:
        if hamer_weights is None:
            raise ValueError("--hamer_weights is required when --run_hamer is set")
        mano_assets_root = os.path.join(hamer_weights, "_DATA", "data")

        # masks_to_hands has its own per-frame skip logic, so always invoke
        # it — re-runs short-circuit cheaply per JSON.
        if not _step("HaMeR per-frame regression", False):
            run_masks_to_hands(
                frames_dir       = frames_dir,
                masks_dir        = masks_dir,
                tracks_path      = hand_tracks,
                output_dir       = hamer_dir,
                weights_dir      = hamer_weights,
                bbox_expansion   = bbox_expansion,
                mask_min_pixels  = mask_min_pixels,
                dev              = dev,
            )

        if not _step("Render HaMeR mesh overlay", os.path.exists(hamer_overlay)):
            run_render_hands_video(
                frames_dir       = frames_dir,
                hamer_dir        = hamer_dir,
                mano_assets_root = mano_assets_root,
                output_path      = hamer_overlay,
                dev              = dev,
            )

        # Align HaMeR to MoGe depth + real intrinsics. Per-frame skip is
        # internal, so always invoke. When the object is tracked, its mask
        # excludes object pixels from the depth-comparison region.
        object_masks_dir = (
            f"{masks_dir}/{object_track_id}"
            if (object_prompt is not None and object_track_id is not None) else None
        )
        if not _step("Align HaMeR to depth", False):
            run_align_hands(
                hamer_dir         = hamer_dir,
                depth_dir         = depth_dir,
                intrinsics_path   = intrinsics_stable,
                mano_assets_root  = mano_assets_root,
                output_dir        = hamer_aligned_dir,
                hand_masks_dir    = masks_dir,
                object_masks_dir  = object_masks_dir,
                mask_min_pixels   = mask_min_pixels,
                dev               = dev,
            )

        # Aligned overlay — pass object mesh + smoothed poses if available.
        object_mesh_arg  = mesh_scaled       if (object_prompt is not None) else None
        object_poses_arg = poses_smooth_dir  if (object_prompt is not None) else None
        if not _step("Render aligned HaMeR overlay",
                     os.path.exists(hamer_aligned_overlay)):
            run_render_hands_aligned_video(
                frames_dir         = frames_dir,
                aligned_dir        = hamer_aligned_dir,
                mano_assets_root   = mano_assets_root,
                output_path        = hamer_aligned_overlay,
                object_mesh_path   = object_mesh_arg,
                object_poses_dir   = object_poses_arg,
                dev                = dev,
            )

    # 8. Optional joint hand+object refinement via Gaussian splatting --------
    # Requires HaMeR aligned outputs (run_hamer) AND a tracked object
    # (object_prompt). Resolves left/right tracks from hand_tracks.json.
    refined_poses_dir          = f"{output_dir}/poses_refined"
    refined_hamer_dir          = f"{output_dir}/hamer_refined"
    refined_overlay            = f"{output_dir}/refined_overlay.mp4"
    hamer_refined_overlay      = f"{output_dir}/hamer_refined_overlay.mp4"
    refined_object_scale_json  = f"{output_dir}/refined_object_scale.json"
    refine_checkpoint_pt       = f"{output_dir}/refine_checkpoint.pt"
    # Optional second-pass refinement output paths (resume from pass1's
    # checkpoint, lower LRs, larger batch).
    refined_overlay_pass2      = f"{output_dir}/refined_overlay_pass2.mp4"
    refine_checkpoint_pt_pass2 = f"{output_dir}/refine_checkpoint_pass2.pt"
    if run_refinement:
        if not (run_hamer and object_prompt is not None):
            raise ValueError(
                "--run_refinement requires both --run_hamer and --object_prompt"
            )

        with open(hand_tracks) as f:
            htm = json.load(f)["tracks"]
        left_tid: int | None  = next((t["object_id"] for t in htm if not t["is_right"]), None)
        right_tid: int | None = next((t["object_id"] for t in htm if     t["is_right"]), None)

        def _maybe(d: str | None) -> str | None:
            return d if d is not None else None

        left_pose_in   = f"{hamer_aligned_dir}/{left_tid}"  if left_tid  is not None else None
        right_pose_in  = f"{hamer_aligned_dir}/{right_tid}" if right_tid is not None else None
        left_mask_in   = f"{masks_dir}/{left_tid}"          if left_tid  is not None else None
        right_mask_in  = f"{masks_dir}/{right_tid}"         if right_tid is not None else None
        left_pose_out  = f"{refined_hamer_dir}/{left_tid}"  if left_tid  is not None else None
        right_pose_out = f"{refined_hamer_dir}/{right_tid}" if right_tid is not None else None

        # All pass-1 kwargs collected in a dict so pass-2 can reuse the
        # base config with just a few overrides instead of duplicating
        # the whole call.
        pass1_kwargs: dict = dict(
                frames_dir                  = frames_dir,
                intrinsics_path             = intrinsics_stable,
                object_mesh_path            = mesh_scaled,
                object_poses_dir            = poses_dir,
                object_mask_dir             = f"{masks_dir}/{object_track_id}",
                refined_object_poses_dir    = refined_poses_dir,
                overlay_path                = refined_overlay,
                refined_object_scale_path   = refined_object_scale_json,
                # Save full state at end so the run can be extended via
                # --refinement_resume on a follow-up invocation.
                checkpoint_path             = refine_checkpoint_pt,
                # If --refinement_resume was passed and the previous run's
                # checkpoint exists, pick up from there. Otherwise start fresh.
                resume_from_checkpoint      = (
                    refine_checkpoint_pt
                    if refinement_resume and os.path.exists(refine_checkpoint_pt)
                    else None
                ),
                # Object-pose-only mode: freeze Gaussians, hand pose, and
                # background pose. Typically combined with --refinement_resume
                # so we start from a settled scene and only nudge obj pose.
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
                mano_assets_root            = mano_assets_root,
                n_epochs                    = 64,
                batch_size                  = 32,
                render_every                = refinement_render_every,
                lr_gaussians                = 3e-2,
                lr_hand_gaussians           = 3e-2,
                # Per-attribute LR multipliers (applied on top of the
                # per-set base LRs above). Roughly the standard 3DGS
                # ratios — positions sensitive, opacity loose, color
                # moderate — adapted to our base LR of 1e-2.
                lr_mul_delta_p              = 0.1,
                lr_mul_quat                 = 0.5,
                lr_mul_scale                = 3.0,
                lr_mul_opacity              = 5.0,
                lr_mul_color                = 1.0,
                lr_mul_obj_global_scale     = 1.0,
                lr_object_pose              = 3e-3,    # legacy lumped LR (used when specifics are None)
                # Per-DOF object pose LRs — rotation typically needs a
                # bump because its gradient is ~r× smaller than
                # translation's (where r = object radius) due to leverage.
                # Try 3-5× translation if rotation converges too slowly.
                lr_object_rot               = 1e-1,
                lr_object_trans             = 3e-3,
                lr_hand_pose                = 3e-3,    # legacy lumped LR (used when specifics are None)
                # Per-DOF hand pose LRs — global_orient and cam_t are the
                # primary "where is the hand" knobs; hand_pose (finger
                # articulation) is finickier so usually lower.
                lr_hand_global_orient       = 1e-2,
                lr_hand_finger              = 3e-3,
                lr_hand_trans               = 3e-3,
                lr_betas                    = 1e-4,
                w_photometric               = 1.0,
                w_silhouette                 = 1.0,
                w_silhouette_hand            = 1.0,
                w_silhouette_obj            = 1.0,
                w_depth                     = 0.001,
                w_smooth_obj_rot            = 0.001,
                w_smooth_obj_trans          = 0.001,
                w_smooth_hand_rot           = 0.001,
                w_smooth_hand_finger        = 0.0001,
                w_smooth_hand_trans         = 0.001,
                w_beta_prior                = 0.1,
                w_obj_scale_prior           = 1.0,
                n_gaussian_only_epochs      = 2,
                seed                        = 0,
                with_background             = True,
                mask_background_to_black    = False,
                balance_photometric_by_mask = False,
                bg_ref_frame                = reference_frame,
                lr_bg_gaussians             = 3e-2,
                lr_bg_pose                  = 3e-3,    # legacy lumped LR (used when specifics are None)
                # Per-DOF bg pose LRs — typically rotation needs a small
                # bump for the same leverage reason as object pose.
                lr_bg_rot                   = 1e-2,
                lr_bg_trans                 = 3e-3,
                use_cosine_lr_schedule      = True,
                cosine_lr_min_ratio         = 0.1,
                coarse_init_scale_factor    = 1.0,
                pose_confidence_decay       = 0.0,
                pose_confidence_ref_frame   = reference_frame,
                w_pose_init_prior           = 0.0,
                # Discrete rotation search to recover frames whose
                # FoundationPose rotation is outside the photometric
                # basin. Runs once at the warmup boundary by default.
                rotation_search_n_candidates      = 0,
                rotation_search_period            = 0,    # 0 = once at warmup boundary; >0 = every K epochs
                rotation_search_local_frac        = 0.5,
                rotation_search_local_max_deg     = 15.0,
                rotation_search_silhouette_weight = 1.0,
                rotation_search_smoothness_weight = 1.0,    # causal: penalize candidates that diverge from previous (snapped) frame
                use_l2_photometric               = False,
                use_l2_silhouette                = False,
                pose_confidence_dynamic_tau      = 0.0,
                train_resolution_scale          = 0.5,
                n_obj_gaussians=5000,
                bg_max_points=20000,
                dev                         = dev,
        )
        if not _step("Gaussian-splat refinement (pass 1)",
                     os.path.exists(refined_overlay)):
            run_refine(**pass1_kwargs)

        # Optional second pass: resume from pass-1's checkpoint, lower
        # LRs (default 0.3×), larger batch, no warmup. Adam state is
        # reinitialized because the new LRs differ; the lowered LR keeps
        # the optimizer in a "polishing" regime without re-introducing
        # the large per-step motion that the original LRs allow.
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
                # Schedule / batching.
                "n_epochs":                sp_n,
                "batch_size":              sp_bs,
                "n_gaussian_only_epochs":  0,
                # Resume from pass-1 state; new LRs ⇒ wipe Adam moments.
                "checkpoint_path":         refine_checkpoint_pt_pass2,
                "resume_from_checkpoint":  refine_checkpoint_pt,
                "ignore_optimizer_state":  True,
                "overlay_path":            refined_overlay_pass2,
                # Random init / pose-only modes only make sense on pass 1.
                "random_init_obj_pose":    False,
                # Scale every LR by sp_lr.
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

        # Render the refined HaMeR + object meshes the same way the
        # aligned overlay does — but using the refined poses + refined
        # MANO params. Output is a directly comparable video to
        # hamer_aligned_overlay.mp4 so you can A/B the refinement.
        # Read the learned object scale (saved by run_refine) and apply
        # it explicitly to the mesh; per-frame Transform3d.scale stays
        # at whatever FoundationPose tracked.
        learned_object_scale = 1.0
        if os.path.exists(refined_object_scale_json):
            with open(refined_object_scale_json) as f:
                learned_object_scale = float(json.load(f).get("scale", 1.0))
            print(f"  Loaded learned object scale: {learned_object_scale:.4f}")
        if not _step("Render refined HaMeR overlay",
                     os.path.exists(hamer_refined_overlay)):
            run_render_hands_aligned_video(
                frames_dir         = frames_dir,
                aligned_dir        = refined_hamer_dir,
                mano_assets_root   = mano_assets_root,
                output_path        = hamer_refined_overlay,
                object_mesh_path   = mesh_scaled,
                object_poses_dir   = refined_poses_dir,
                object_scale       = learned_object_scale,
                dev                = dev,
            )

        # Plot per-frame pose-gradient RMS from the saved checkpoint
        # (useful for spotting "uncertain" frames where the optimizer
        # was still actively pushing the pose at end of training).
        _plot_pose_grad_from_checkpoint(
            refine_checkpoint_pt,
            f"{output_dir}/pose_grad.png",
        )

    print(f"\n{'='*60}")
    print(f"  Done.")
    if undistort:
        print(f"  anycalib        : {anycalib_dir}/")
    print(f"  hand_detections : {hand_detections}")
    print(f"  hand_tracks     : {hand_tracks}")
    print(f"  masks           : {masks_dir}/{{1..{len(detections)}}}/")
    print(f"  prompts_overlay : {prompts_overlay}")
    print(f"  masks_overlay   : {masks_overlay}")
    if run_hamer:
        print(f"  depth           : {depth_dir}/")
        print(f"  intrinsics      : {intrinsics_stable}")
        print(f"  hamer           : {hamer_dir}/{{1..{len(detections)}}}/*.json")
        print(f"  hamer_overlay   : {hamer_overlay}")
        print(f"  hamer_aligned   : {hamer_aligned_dir}/{{1..{len(detections)}}}/*.json")
        print(f"  hamer_aligned_overlay : {hamer_aligned_overlay}")
    if run_refinement:
        print(f"  poses_refined         : {refined_poses_dir}/")
        print(f"  hamer_refined         : {refined_hamer_dir}/")
        print(f"  refined_overlay       : {refined_overlay}")
        print(f"  hamer_refined_overlay : {hamer_refined_overlay}")
    print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_path",   required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--sam2_weights", default="data/weights/sam2")
    p.add_argument("--reference_frame", type=int, default=0)
    p.add_argument("--selfie", action="store_true",
                   help="Input is from a selfie/front-facing camera (mirrored). "
                        "Defaults to rear/ego convention.")
    p.add_argument("--max_num_hands", type=int, default=2)
    p.add_argument("--min_detection_confidence", type=float, default=0.1,
                   help="MediaPipe detection threshold (default 0.3 — lowered "
                        "from MediaPipe's stock 0.5 for harder views).")
    p.add_argument("--pad_ratio", type=float, default=0.15)
    p.add_argument("--sam2_bbox_pad", type=float, default=0.2,
                   help="Pad each hand-detection bbox by this ratio AND "
                        "square it before passing to SAM2 (e.g. 0.2 = "
                        "20%% margin on the longer side, then squared). "
                        "Set to 0 to use raw MediaPipe bboxes unchanged.")
    p.add_argument("--undistort", action="store_true",
                   help="Run AnyCalib first to estimate intrinsics + distortion "
                        "and rebind video_path to the undistorted MP4.")
    p.add_argument("--anycalib_weights", default="data/weights/anycalib",
                   help="AnyCalib weights dir (used only with --undistort).")
    p.add_argument("--moge_weights", default="data/weights/moge",
                   help="MoGe weights dir (used only with --run_hamer).")
    p.add_argument("--run_hamer", action="store_true",
                   help="After SAM2, run HaMeR per (track, frame), align it "
                        "to MoGe depth, and render verification overlays.")
    p.add_argument("--hamer_weights", default="data/weights/hamer",
                   help="Dir containing _DATA/hamer_ckpts/ (and MANO_RIGHT.pkl "
                        "under _DATA/data/).")
    p.add_argument("--bbox_expansion", type=float, default=1.7,
                   help="Expand SAM2 mask bbox by this factor for HaMeR crop.")
    p.add_argument("--mask_min_pixels", type=int, default=256,
                   help="Skip HaMeR when SAM2 mask has fewer than this many pixels.")
    p.add_argument("--object_prompt", default=None,
                   help="Grounding-DINO text prompt for the held object "
                        "(e.g. 'blue cup'). When set, runs the full object "
                        "branch: DINO → SAM2 → SAM3D → FoundationPose.")
    p.add_argument("--grounding_dino_weights",  default="data/weights/grounding_dino")
    p.add_argument("--sam3d_weights",           default="data/weights/sam3d")
    p.add_argument("--foundation_pose_weights", default="data/weights/foundation_pose")
    p.add_argument("--reregister_iou_thresh", type=float, default=0.3,
                   help="FoundationPose re-register-from-scratch IoU threshold "
                        "(0 to disable).")
    p.add_argument("--run_refinement", action="store_true",
                   help="After alignment, jointly refine hand+object poses via "
                        "Gaussian splatting. Requires --run_hamer and "
                        "--object_prompt.")
    p.add_argument("--refinement_epochs", type=int, default=30,
                   help="Number of refinement epochs (each epoch visits "
                        "every frame once in a random permutation).")
    p.add_argument("--refinement_batch_size", type=int, default=4,
                   help="Frames per refinement optimizer step.")
    p.add_argument("--refinement_resume", action="store_true",
                   help="Resume gsplat refinement from "
                        "<output_dir>/refine_checkpoint.pt if present. "
                        "Subsequent invocations will continue training "
                        "from saved state instead of starting fresh.")
    p.add_argument("--refinement_object_pose_only", action="store_true",
                   help="Freeze Gaussians + hand pose + bg pose; only "
                        "the object pose updates. Useful after an initial "
                        "training pass to fine-tune object pose alone "
                        "(typically combined with --refinement_resume).")
    p.add_argument("--refinement_random_init_obj_pose", action="store_true",
                   help="Randomize per-frame object pose (uniform SO(3) "
                        "rotation, σ=0.1m translation noise) before "
                        "training. Tests whether the optimizer can "
                        "recover the pose from scratch.")
    p.add_argument("--refinement_ignore_optimizer_state", action="store_true",
                   help="When resuming, skip loading optimizer/LR-scheduler "
                        "state — Adam moments reinit lazily. Useful when "
                        "changing LRs / loss weights between runs.")
    p.add_argument("--refinement_second_pass", action="store_true",
                   help="After the first refinement, run a second polishing "
                        "pass: resumes from pass-1's checkpoint, lower LRs "
                        "(by --refinement_second_pass_lr_scale), larger "
                        "batch, no warmup. Adam state is reinitialized so "
                        "the new LRs aren't fought by stale moments.")
    p.add_argument("--refinement_second_pass_lr_scale", type=float, default=0.3,
                   help="Multiplier applied to every LR in pass 2 "
                        "(Gaussian + pose + bg). 0.3 = polish; 0.1 = "
                        "very fine polish; 1.0 = same LR as pass 1.")
    p.add_argument("--refinement_second_pass_epochs", type=int, default=None,
                   help="Number of pass-2 epochs. Defaults to half of "
                        "pass-1's n_epochs (min 8).")
    p.add_argument("--refinement_second_pass_batch_size", type=int, default=None,
                   help="Batch size for pass 2. Defaults to 2x pass-1.")
    p.add_argument("--refinement_render_every", type=int, default=25,
                   help="Dump an overlay PNG of a reference frame every N "
                        "optimizer steps during refinement (0 disables). "
                        "Stitched into progress.mp4 at the end.")
    p.add_argument("--dev", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_hand_masks(
        video_path               = args.video_path,
        output_dir               = args.output_dir,
        sam2_weights             = args.sam2_weights,
        reference_frame          = args.reference_frame,
        selfie                   = args.selfie,
        max_num_hands            = args.max_num_hands,
        min_detection_confidence = args.min_detection_confidence,
        pad_ratio                = args.pad_ratio,
        sam2_bbox_pad            = args.sam2_bbox_pad,
        undistort                = args.undistort,
        anycalib_weights         = args.anycalib_weights,
        moge_weights             = args.moge_weights,
        run_hamer                = args.run_hamer,
        hamer_weights            = args.hamer_weights,
        bbox_expansion           = args.bbox_expansion,
        mask_min_pixels          = args.mask_min_pixels,
        object_prompt            = args.object_prompt,
        grounding_dino_weights   = args.grounding_dino_weights,
        sam3d_weights            = args.sam3d_weights,
        foundation_pose_weights  = args.foundation_pose_weights,
        reregister_iou_thresh    = args.reregister_iou_thresh,
        run_refinement           = args.run_refinement,
        refinement_epochs        = args.refinement_epochs,
        refinement_batch_size    = args.refinement_batch_size,
        refinement_render_every  = args.refinement_render_every,
        refinement_resume                  = args.refinement_resume,
        refinement_object_pose_only        = args.refinement_object_pose_only,
        refinement_random_init_obj_pose    = args.refinement_random_init_obj_pose,
        refinement_ignore_optimizer_state  = args.refinement_ignore_optimizer_state,
        refinement_second_pass             = args.refinement_second_pass,
        refinement_second_pass_lr_scale    = args.refinement_second_pass_lr_scale,
        refinement_second_pass_epochs      = args.refinement_second_pass_epochs,
        refinement_second_pass_batch_size  = args.refinement_second_pass_batch_size,
        dev                      = args.dev,
    )
