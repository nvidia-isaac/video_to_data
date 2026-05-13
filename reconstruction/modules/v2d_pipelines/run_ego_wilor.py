"""WiLoR-driven hand pipeline: detect per frame → SAM2 → IoU match → per-track JSONs.

Differs from ``run_hand_masks.py`` in two ways:
  * the hand detector is WiLoR (per-frame, end-to-end bbox + MANO) rather
    than MediaPipe on the reference frame only;
  * per-track per-frame MANO assignment is done by rendering each WiLoR
    detection's MANO mesh and IoU-matching it against the propagated SAM2
    masks, rather than running a separate pose model on mask-cropped patches.

Steps:
  0.  AnyCalib undistortion           [optional, --undistort]
  1.  Extract frames
  2.  MoGe depth + intrinsics         → depth/ + intrinsics_stable.json
  2c. DROID-SLAM trajectory           [optional, --run_slam]
                                       → slam_poses/<frame:06d>.json
                                       (scale-aligned to MoGe depth; seeds
                                       gsplat's background pose field
                                       at step 15 when --run_refinement)
  3.  WiLoR over all frames           → wilor_raw/<frame:06d>.json
  4.  (Object) Grounding DINO          [only when --object_prompt is set]
  5.  Build SAM2 prompts (object + wilor ref-frame hands)
                                       → sam2_prompts.json + hand_tracks.json
  6.  SAM2 mask propagation            → masks/{1,2,…}/*.png
  7.  Render prompts overlay + masks overlay
  8.  Object branch (SAM3D + FoundationPose + EKF)   [--object_prompt]
  9.  Match wilor detections to SAM2 tracks via silhouette IoU
                                       → wilor/<track_id>/<frame:06d>.json
  10. Render virtual-cam overlay (raw) → wilor_overlay.mp4
  11. Depth-align hands, real frames only (real intrinsics)
                                       → wilor_aligned/<track_id>/<frame:06d>.json
  12. Render aligned overlay (+ object if --object_prompt)
                                       → wilor_aligned_overlay.mp4
  13. Interpolate missing aligned frames
                                       → wilor_aligned_filled/<track_id>/<frame:06d>.json
                                         (tagged `interpolated: true|false`)
  14. Render filled aligned overlay     → wilor_aligned_filled_overlay.mp4
  15. Joint hand+object GS refinement   [--run_refinement, requires --object_prompt]
                                       → poses_refined/, wilor_refined/, *_overlay.mp4

Output directory layout:
  <output_dir>/
  ├── frames/
  ├── depth/                           # MoGe depth PNGs
  ├── intrinsics/                      # MoGe per-frame intrinsics JSONs
  ├── intrinsics_stable.json
  ├── slam_poses/<frame:06d>.json      # [--run_slam] DROID-SLAM cam-to-world
  ├── slam_trajectory.txt              # [--run_slam] TUM format
  ├── wilor_raw/<frame:06d>.json
  ├── hand_detections.json             # ref-frame slice of wilor_raw
  ├── sam2_prompts.json
  ├── hand_tracks.json
  ├── masks/{...}/*.png
  ├── prompts_overlay.png
  ├── masks_overlay.mp4
  ├── wilor/{...}/<frame:06d>.json                  # real detections only
  ├── wilor_overlay.mp4
  ├── wilor_aligned/{...}/<frame:06d>.json          # aligned real frames
  ├── wilor_aligned_overlay.mp4
  ├── wilor_aligned_filled/{...}/<frame:06d>.json   # real + interpolated (tagged)
  └── wilor_aligned_filled_overlay.mp4

Usage:
    python modules/v2d_pipelines/run_ego_wilor.py \\
        --video_path data/clip.mp4 \\
        --output_dir data/outputs/clip \\
        --sam2_weights  data/weights/sam2 \\
        --wilor_weights data/weights/wilor \\
        --undistort

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


from v2d.anycalib.docker.run_video_to_calibration import run_video_to_calibration
from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.common.utils import extract_images
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.droid_slam.docker.run_video_to_slam import run_video_to_slam
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.gsplat_refinement.docker.run_refine import run_refine
from v2d.hamer.docker.run_align_hands import run_align_hands
from v2d.hamer.docker.run_render_hands_aligned_video import run_render_hands_aligned_video
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_moge_depth
from v2d.pipelines.run_hand_masks import _plot_pose_grad_from_checkpoint
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
from v2d.wilor.docker.run_render_hands_video import run_render_hands_video as run_wilor_render
from v2d.wilor.docker.run_tracks_from_wilor_masks import run_tracks_from_wilor_masks
from v2d.wilor.docker.run_tracks_interpolate import run_tracks_interpolate
from v2d.wilor.docker.run_video_to_hands import run_video_to_hands as run_wilor_video


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


def _pad_and_square_bbox(
    bbox: dict, pad_ratio: float, img_w: int, img_h: int,
) -> dict:
    """Center, square at the longer side x (1+pad_ratio), clamp to image."""
    x0, y0, x1, y1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * (1.0 + pad_ratio)
    half = side / 2.0
    return {
        "x0": max(0.0,          cx - half),
        "y0": max(0.0,          cy - half),
        "x1": min(float(img_w), cx + half),
        "y1": min(float(img_h), cy + half),
    }


def run_ego_wilor(
    video_path: str,
    output_dir: str,
    sam2_weights: str,
    wilor_weights: str = "data/weights/wilor",
    reference_frame: int = 0,
    sam2_bbox_pad: float = 0.2,
    undistort: bool = False,
    anycalib_weights: str = "data/weights/anycalib",
    moge_weights: str = "data/weights/moge",
    min_iou: float = 0.1,
    mask_min_pixels: int = 256,
    interp_betas: str = "fixed",
    interp_max_gap_frames: int = 15,
    object_prompt: str | None = None,
    grounding_dino_weights: str = "data/weights/grounding_dino",
    sam3d_weights: str = "data/weights/sam3d",
    foundation_pose_weights: str = "data/weights/foundation_pose",
    reregister_iou_thresh: float | None = 0.3,
    run_slam: bool = False,
    droid_slam_weights: str = "data/weights/droid_slam",
    run_refinement: bool = False,
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
    wilor_raw_dir      = f"{output_dir}/wilor_raw"
    hand_detections    = f"{output_dir}/hand_detections.json"
    dino_detections    = f"{output_dir}/dino_detections.json"
    sam2_prompts       = f"{output_dir}/sam2_prompts.json"
    hand_tracks        = f"{output_dir}/hand_tracks.json"
    object_track       = f"{output_dir}/object_track.json"
    masks_dir          = f"{output_dir}/masks"
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
    poses_dir          = f"{output_dir}/poses"
    poses_smooth_dir   = f"{output_dir}/poses_smoothed"
    slam_poses_dir            = f"{output_dir}/slam_poses"
    slam_trajectory           = f"{output_dir}/slam_trajectory.txt"
    # Hand outputs
    wilor_tracks_dir          = f"{output_dir}/wilor"
    wilor_overlay             = f"{output_dir}/wilor_overlay.mp4"
    wilor_aligned_dir         = f"{output_dir}/wilor_aligned"
    wilor_aligned_overlay     = f"{output_dir}/wilor_aligned_overlay.mp4"
    wilor_aligned_filled_dir  = f"{output_dir}/wilor_aligned_filled"
    wilor_aligned_filled_overlay = f"{output_dir}/wilor_aligned_filled_overlay.mp4"
    ref_rgb            = f"{frames_dir}/{reference_frame:06d}.png"
    ref_depth          = f"{depth_dir}/{reference_frame:06d}.png"

    print(f"\n{'='*60}")
    print(f"  video           : {os.path.basename(video_path)}")
    print(f"  output          : {output_dir}")
    print(f"  reference_frame : {reference_frame}")
    print(f"  undistort       : {undistort}")
    print(f"  object_prompt   : {object_prompt}")
    print(f"  run_slam        : {run_slam}")
    print(f"  min_iou         : {min_iou}")
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

    # 1. Extract frames ------------------------------------------------------
    if not _step("Extract frames", _has_files(frames_dir)):
        extract_images(video_path, frames_dir)

    # 2. MoGe depth + intrinsics --------------------------------------------
    # Needed for both the depth-alignment step (hands) and, when set, the
    # object branch (SAM3D + FoundationPose).
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
            det["bbox"] = _pad_and_square_bbox(
                det["bbox"], sam2_bbox_pad, ref_w, ref_h,
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

    # 6. SAM2 propagation ----------------------------------------------------
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

    # 8. Object branch: SAM3D + FoundationPose -------------------------------
    if object_prompt is not None:
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

    # 9. IoU-match wilor detections to SAM2 hand tracks ----------------------
    mano_assets_root = os.path.join(wilor_weights, "pretrained_models")
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

    # 10. Render virtual-cam verification overlay (raw, real detections only)
    if not _step("Render wilor mesh overlay (raw)", os.path.exists(wilor_overlay)):
        run_wilor_render(
            frames_dir       = frames_dir,
            wilor_dir        = wilor_tracks_dir,
            mano_assets_root = mano_assets_root,
            output_path      = wilor_overlay,
            dev              = dev,
        )

    # 11. Depth alignment to MoGe + real intrinsics (real frames only) -----
    # Run alignment on the real-only matched detections. Interpolated frames
    # would lie on a guessed pose that may not match the image silhouette, so
    # aligning them against real depth would corrupt cam_t. We interpolate
    # AFTER alignment (step 13).
    # Reuses hamer's run_align_hands — schema is hamer-compatible.
    object_masks_dir = (
        f"{masks_dir}/{object_track_id}"
        if (object_prompt is not None and object_track_id is not None) else None
    )
    if not _step("Align wilor hands to depth (real frames only)", False):
        run_align_hands(
            hamer_dir         = wilor_tracks_dir,
            depth_dir         = depth_dir,
            intrinsics_path   = intrinsics_stable,
            mano_assets_root  = mano_assets_root,
            output_dir        = wilor_aligned_dir,
            hand_masks_dir    = masks_dir,
            object_masks_dir  = object_masks_dir,
            mask_min_pixels   = mask_min_pixels,
            dev               = dev,
        )

    # 12. Render aligned overlay (real frames only) -------------------------
    object_mesh_arg  = mesh_scaled       if (object_prompt is not None) else None
    object_poses_arg = poses_smooth_dir  if (object_prompt is not None) else None
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
    aligned_filled_done = any(
        _has_files(os.path.join(wilor_aligned_filled_dir, d))
        for d in (os.listdir(wilor_aligned_filled_dir)
                  if os.path.isdir(wilor_aligned_filled_dir) else [])
    )
    if not _step("Interpolate missing aligned frames", aligned_filled_done):
        run_tracks_interpolate(
            aligned_dir    = wilor_aligned_dir,
            masks_dir      = masks_dir,
            output_dir     = wilor_aligned_filled_dir,
            betas          = interp_betas,
            max_gap_frames = interp_max_gap_frames,
            dev            = dev,
        )

    # 14. Render filled aligned overlay (real + interpolated) --------------
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

    # 15. Optional joint hand+object refinement via Gaussian splatting -----
    # Mirrors run_hand_masks.py's refinement block; sources hand poses from
    # wilor_aligned_filled/ (real + interpolated, hamer-compatible schema).
    # gsplat sees one record per visible frame and can re-optimize the
    # interpolated frames against the image.
    refined_poses_dir          = f"{output_dir}/poses_refined"
    refined_wilor_dir          = f"{output_dir}/wilor_refined"
    refined_overlay            = f"{output_dir}/refined_overlay.mp4"
    wilor_refined_overlay      = f"{output_dir}/wilor_refined_overlay.mp4"
    refined_object_scale_json  = f"{output_dir}/refined_object_scale.json"
    refine_checkpoint_pt       = f"{output_dir}/refine_checkpoint.pt"
    refined_overlay_pass2      = f"{output_dir}/refined_overlay_pass2.mp4"
    refine_checkpoint_pt_pass2 = f"{output_dir}/refine_checkpoint_pass2.pt"
    if run_refinement:
        if object_prompt is None:
            raise ValueError(
                "--run_refinement requires --object_prompt (joint hand+object refinement)."
            )

        with open(hand_tracks) as f:
            htm = json.load(f)["tracks"]
        left_tid: int | None  = next((t["object_id"] for t in htm if not t["is_right"]), None)
        right_tid: int | None = next((t["object_id"] for t in htm if     t["is_right"]), None)

        def _maybe(d: str | None) -> str | None:
            return d if d is not None else None

        left_pose_in   = f"{wilor_aligned_filled_dir}/{left_tid}"  if left_tid  is not None else None
        right_pose_in  = f"{wilor_aligned_filled_dir}/{right_tid}" if right_tid is not None else None
        left_mask_in   = f"{masks_dir}/{left_tid}"          if left_tid  is not None else None
        right_mask_in  = f"{masks_dir}/{right_tid}"         if right_tid is not None else None
        left_pose_out  = f"{refined_wilor_dir}/{left_tid}"  if left_tid  is not None else None
        right_pose_out = f"{refined_wilor_dir}/{right_tid}" if right_tid is not None else None

        pass1_kwargs: dict = dict(
                frames_dir                  = frames_dir,
                intrinsics_path             = intrinsics_stable,
                object_mesh_path            = mesh_scaled,
                object_poses_dir            = poses_dir,
                object_mask_dir             = f"{masks_dir}/{object_track_id}",
                refined_object_poses_dir    = refined_poses_dir,
                overlay_path                = refined_overlay,
                refined_object_scale_path   = refined_object_scale_json,
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
                mano_assets_root            = mano_assets_root,
                n_epochs                    = 64,
                batch_size                  = 32,
                render_every                = refinement_render_every,
                lr_gaussians                = 3e-2,
                lr_hand_gaussians           = 3e-2,
                lr_mul_delta_p              = 0.1,
                lr_mul_quat                 = 0.5,
                lr_mul_scale                = 3.0,
                lr_mul_opacity              = 5.0,
                lr_mul_color                = 1.0,
                lr_mul_obj_global_scale     = 1.0,
                lr_object_pose              = 3e-3,
                lr_object_rot               = 1e-1,
                lr_object_trans             = 3e-3,
                lr_hand_pose                = 3e-3,
                lr_hand_global_orient       = 1e-2,
                lr_hand_finger              = 3e-3,
                lr_hand_trans               = 3e-3,
                lr_betas                    = 1e-4,
                w_photometric               = 1.0,
                w_silhouette                = 1.0,
                w_silhouette_hand           = 1.0,
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
                # Seed bg pose from DROID-SLAM (when --run_slam). Without
                # this, BackgroundPoseField identity-inits, which is a tiny
                # basin for moving cameras.
                background_pose_init_dir    = slam_poses_dir if run_slam else None,
                mask_background_to_black    = False,
                balance_photometric_by_mask = False,
                bg_ref_frame                = reference_frame,
                lr_bg_gaussians             = 3e-2,
                lr_bg_pose                  = 3e-3,
                lr_bg_rot                   = 1e-2,
                lr_bg_trans                 = 3e-3,
                use_cosine_lr_schedule      = True,
                cosine_lr_min_ratio         = 0.1,
                coarse_init_scale_factor    = 1.0,
                pose_confidence_decay       = 0.0,
                pose_confidence_ref_frame   = reference_frame,
                w_pose_init_prior           = 0.0,
                rotation_search_n_candidates      = 0,
                rotation_search_period            = 0,
                rotation_search_local_frac        = 0.5,
                rotation_search_local_max_deg     = 15.0,
                rotation_search_silhouette_weight = 1.0,
                rotation_search_smoothness_weight = 1.0,
                use_l2_photometric                = False,
                use_l2_silhouette                 = False,
                pose_confidence_dynamic_tau       = 0.0,
                train_resolution_scale            = 0.5,
                n_obj_gaussians                   = 5000,
                bg_max_points                     = 20000,
                dev                         = dev,
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
        if os.path.exists(refined_object_scale_json):
            with open(refined_object_scale_json) as f:
                learned_object_scale = float(json.load(f).get("scale", 1.0))
            print(f"  Loaded learned object scale: {learned_object_scale:.4f}")
        if not _step("Render refined wilor overlay",
                     os.path.exists(wilor_refined_overlay)):
            run_render_hands_aligned_video(
                frames_dir         = frames_dir,
                aligned_dir        = refined_wilor_dir,
                mano_assets_root   = mano_assets_root,
                output_path        = wilor_refined_overlay,
                object_mesh_path   = mesh_scaled,
                object_poses_dir   = refined_poses_dir,
                object_scale       = learned_object_scale,
                dev                = dev,
            )

        _plot_pose_grad_from_checkpoint(
            refine_checkpoint_pt,
            f"{output_dir}/pose_grad.png",
        )

    print(f"\n{'='*60}")
    print(f"  Done.")
    if undistort:
        print(f"  anycalib        : {anycalib_dir}/")
    print(f"  wilor_raw       : {wilor_raw_dir}/")
    print(f"  hand_detections : {hand_detections}")
    print(f"  hand_tracks     : {hand_tracks}")
    print(f"  masks           : {masks_dir}/")
    print(f"  prompts_overlay : {prompts_overlay}")
    print(f"  masks_overlay   : {masks_overlay}")
    print(f"  wilor (raw tracks)           : {wilor_tracks_dir}/")
    print(f"  wilor_overlay                : {wilor_overlay}")
    print(f"  wilor_aligned                : {wilor_aligned_dir}/  (real only)")
    print(f"  wilor_aligned_overlay        : {wilor_aligned_overlay}")
    print(f"  wilor_aligned_filled         : {wilor_aligned_filled_dir}/  (real + interpolated)")
    print(f"  wilor_aligned_filled_overlay : {wilor_aligned_filled_overlay}")
    if run_slam:
        print(f"  slam_poses                   : {slam_poses_dir}/")
        print(f"  slam_trajectory              : {slam_trajectory}")
    if run_refinement:
        print(f"  poses_refined         : {refined_poses_dir}/")
        print(f"  wilor_refined         : {refined_wilor_dir}/")
        print(f"  refined_overlay       : {refined_overlay}")
        print(f"  wilor_refined_overlay : {wilor_refined_overlay}")
    print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_path",    required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--sam2_weights",  default="data/weights/sam2")
    p.add_argument("--wilor_weights", default="data/weights/wilor")
    p.add_argument("--reference_frame", type=int, default=0)
    p.add_argument("--sam2_bbox_pad", type=float, default=0.2,
                   help="Pad each hand-detection bbox by this ratio AND square it "
                        "before passing to SAM2. 0 = use raw WiLoR bboxes.")
    p.add_argument("--undistort", action="store_true",
                   help="Run AnyCalib first to estimate intrinsics + distortion "
                        "and rebind video_path to the undistorted MP4.")
    p.add_argument("--anycalib_weights", default="data/weights/anycalib")
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
                        "(e.g. 'blue cup'). When set, runs the full object "
                        "branch: DINO → SAM2 → SAM3D → FoundationPose.")
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
    p.add_argument("--run_refinement", action="store_true",
                   help="After alignment, jointly refine hand+object poses via "
                        "Gaussian splatting. Requires --object_prompt.")
    p.add_argument("--refinement_render_every", type=int, default=25,
                   help="Dump an overlay PNG of a reference frame every N "
                        "optimizer steps during refinement (0 disables).")
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


if __name__ == "__main__":
    args = parse_args()
    run_ego_wilor(
        video_path              = args.video_path,
        output_dir              = args.output_dir,
        sam2_weights            = args.sam2_weights,
        wilor_weights           = args.wilor_weights,
        reference_frame         = args.reference_frame,
        sam2_bbox_pad           = args.sam2_bbox_pad,
        undistort               = args.undistort,
        anycalib_weights        = args.anycalib_weights,
        moge_weights            = args.moge_weights,
        min_iou                 = args.min_iou,
        mask_min_pixels         = args.mask_min_pixels,
        interp_betas            = args.interp_betas,
        interp_max_gap_frames   = args.interp_max_gap_frames,
        object_prompt           = args.object_prompt,
        grounding_dino_weights  = args.grounding_dino_weights,
        sam3d_weights           = args.sam3d_weights,
        foundation_pose_weights = args.foundation_pose_weights,
        reregister_iou_thresh   = args.reregister_iou_thresh,
        run_slam                = args.run_slam,
        droid_slam_weights      = args.droid_slam_weights,
        run_refinement                    = args.run_refinement,
        refinement_render_every           = args.refinement_render_every,
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
