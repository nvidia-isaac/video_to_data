"""
Pipeline: Pre-made Mesh + Video → Track → Render

Skips 3D reconstruction — use this when you already have a mesh for the object.
The mesh is assumed to be in metric scale (e.g. from BundleSDF).

Steps:
  1.  Extract frames       — video → frames/
  2.  Grounding DINO       — reference frame + text prompt → bounding box detection
  3.  SAM2                 — video + prompt → per-object masks/
  4.  Depth estimation     — video → depth/ + intrinsics/  (moge / unidepth / depth_anything)
  5.  Stabilise intrinsics — median per-frame focal lengths → intrinsics_stable.json
  6.  Align depth          — per-frame scale correction to reference frame (optional)
  7.  Simplify mesh        — reduce polygon count (optional)
  8.  Estimate scale       — coarse-to-fine grid search to align mesh to depth (optional)
  9.  Track poses          — FoundationPose: video + depth + masks + mesh → poses/
  9b. Correct depth        — FP-guided depth scale correction + re-track (optional)
  10. EKF smoothing        — ESKF + RTS smoother → poses_smoothed/
  11. Render overlays      — mesh + poses + frames → renders/
  12. Encode videos        — renders.mp4 + masks.mp4 + depth.mp4 → comparison.mp4

Each step checks for cached outputs and skips if already complete.

Run from reconstruction/:
    python -m v2d.pipelines.run_premade_mesh_tracking \\
        --video_path  <video> \\
        --mesh_path   <mesh> \\
        --output_dir  <dir> \\
        --detection_prompt "yellow spray can" \\
        --depth_model moge \\
        --depth_weights data/weights/moge \\
        --sam2_weights  data/weights/sam2 \\
        --fp_weights    data/weights/foundation_pose \\
        --dino_weights  data/weights/grounding_dino
"""

import json
import os
from typing import Literal

from v2d.pipelines.extract_images import extract_images
from v2d.pipelines.frames_to_video import frames_to_video
from v2d.pipelines.stitch_videos import stitch_videos
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.mesh.docker.run_mesh_simplify import run_mesh_simplify
from v2d.foundation_pose.docker.run_render_poses import run_render_poses
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_correct_depth_scale import run_correct_depth_scale
from v2d.depth.lib.align_depth_sequence import align_depth_sequence
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


def _dino_detections_to_sam2_prompts(
    detections_path: str,
    frame_index: int,
    object_id: int,
) -> Sam2Prompts:
    with open(detections_path) as f:
        detections = json.load(f)
    if not detections:
        raise RuntimeError(f"Grounding DINO found no detections in {detections_path}")
    top = detections[0]
    box = BoundingBox.from_dict(top["box"])
    return Sam2Prompts(prompts=[
        Sam2Prompt(frame_index=frame_index, object_id=object_id, box=box)
    ])


def run_premade_mesh_tracking(
    video_path: str,
    mesh_path: str,
    detection_prompt: str,
    output_dir: str,
    sam2_weights: str,
    depth_weights: str,
    fp_weights: str,
    dino_weights: str,
    depth_model: Literal["moge", "unidepth", "depth_anything"] = "moge",
    object_id: int = 1,
    reference_frame: int = 40,
    simplify_factor: float | None = None,
    align_depth: bool = False,
    estimate_scale: bool = True,
    correct_depth: bool = False,
) -> None:
    frames_dir         = f"{output_dir}/frames"
    masks_dir          = f"{output_dir}/masks"
    depth_dir          = f"{output_dir}/depth"
    depth_aligned_dir  = f"{output_dir}/depth_aligned"
    intrinsics_dir     = f"{output_dir}/intrinsics"
    poses_dir          = f"{output_dir}/poses"
    poses_smoothed_dir = f"{output_dir}/poses_smoothed"
    renders_dir        = f"{output_dir}/renders"
    intrinsics_stable  = f"{output_dir}/intrinsics_stable.json"
    dino_detections    = f"{output_dir}/dino_detections.json"
    sam2_prompts_path  = f"{output_dir}/sam2_prompts.json"
    ref                = f"{reference_frame:06d}"

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Extract frames
    # -------------------------------------------------------------------------
    if not _has_files(frames_dir):
        print("Step 1: Extracting frames...")
        extract_images(video_path, frames_dir)
    else:
        print("Step 1: Skipping (frames already extracted)")

    # -------------------------------------------------------------------------
    # Step 2: Grounding DINO detection on reference frame
    # -------------------------------------------------------------------------
    if not os.path.exists(dino_detections):
        print("Step 2: Grounding DINO detection...")
        run_image_to_object_bboxes(
            image_path=f"{frames_dir}/{ref}.png",
            output_path=dino_detections,
            prompt=detection_prompt,
            model_dir=dino_weights,
        )
    else:
        print("Step 2: Skipping (DINO detections cached)")

    if not os.path.exists(sam2_prompts_path):
        prompts = _dino_detections_to_sam2_prompts(dino_detections, reference_frame, object_id)
        with open(sam2_prompts_path, "w") as f:
            json.dump(prompts.to_dict(), f, indent=2)

    # -------------------------------------------------------------------------
    # Step 3: SAM2 segmentation
    # -------------------------------------------------------------------------
    if not _has_files(f"{masks_dir}/{object_id}"):
        print("Step 3: SAM2 segmentation...")
        run_video_to_masks(
            video_path=video_path,
            prompts_path=sam2_prompts_path,
            masks_dir=masks_dir,
            weights_dir=sam2_weights,
        )
    else:
        print("Step 3: Skipping (masks already computed)")

    # -------------------------------------------------------------------------
    # Step 4: Depth estimation
    # -------------------------------------------------------------------------
    if not _has_files(depth_dir):
        print(f"Step 4: Depth estimation ({depth_model})...")
        if depth_model == "moge":
            from v2d.moge.docker.run_video_to_depth import run_video_to_depth
        elif depth_model == "unidepth":
            from v2d.unidepth.docker.run_video_to_depth import run_video_to_depth
        elif depth_model == "depth_anything":
            from v2d.depth_anything.docker.run_video_to_depth import run_video_to_depth
        else:
            raise ValueError(f"Unknown depth_model: {depth_model!r}")
        run_video_to_depth(
            video_path=video_path,
            depth_folder=depth_dir,
            intrinsics_folder=intrinsics_dir,
            weights_path=depth_weights,
        )
    else:
        print("Step 4: Skipping (depth already computed)")

    # -------------------------------------------------------------------------
    # Step 5: Stabilise intrinsics
    # -------------------------------------------------------------------------
    if not os.path.exists(intrinsics_stable):
        print("Step 5: Stabilising intrinsics...")
        stabilize_intrinsics(
            intrinsics_folder=intrinsics_dir,
            output_path=intrinsics_stable,
        )
    else:
        print("Step 5: Skipping (stable intrinsics cached)")

    # -------------------------------------------------------------------------
    # Step 6: Align depth (optional)
    # -------------------------------------------------------------------------
    tracking_depth_dir = depth_dir
    if align_depth:
        if not _has_files(depth_aligned_dir):
            print("Step 6: Aligning depth sequence...")
            align_depth_sequence(
                depth_folder=depth_dir,
                frames_folder=frames_dir,
                masks_folder=f"{masks_dir}/{object_id}",
                output_folder=depth_aligned_dir,
                reference_frame=reference_frame,
            )
        else:
            print("Step 6: Skipping (aligned depth cached)")
        tracking_depth_dir = depth_aligned_dir

    # -------------------------------------------------------------------------
    # Step 7: Simplify mesh (optional)
    # -------------------------------------------------------------------------
    tracking_mesh = mesh_path
    if simplify_factor is not None:
        simplified_mesh = f"{output_dir}/mesh_simplified.obj"
        if not os.path.exists(simplified_mesh):
            print("Step 7: Simplifying mesh...")
            run_mesh_simplify(
                input_mesh_path=mesh_path,
                output_mesh_path=simplified_mesh,
                factor=simplify_factor,
            )
        else:
            print("Step 7: Skipping (simplified mesh cached)")
        tracking_mesh = simplified_mesh

    # -------------------------------------------------------------------------
    # Step 8: Estimate mesh scale (optional)
    # -------------------------------------------------------------------------
    if estimate_scale:
        scaled_mesh = f"{output_dir}/mesh_scaled.obj"
        scale_path  = f"{output_dir}/mesh_scale.json"
        if not os.path.exists(scaled_mesh):
            print("Step 8: Estimating mesh scale...")
            run_estimate_mesh_scale(
                mesh_path=tracking_mesh,
                rgb_path=f"{frames_dir}/{ref}.png",
                depth_path=f"{tracking_depth_dir}/{ref}.png",
                mask_path=f"{masks_dir}/{object_id}/{ref}.png",
                intrinsics_path=intrinsics_stable,
                weights_dir=fp_weights,
                scale_path=scale_path,
                rescaled_mesh_path=scaled_mesh,
            )
        else:
            print("Step 8: Skipping (scaled mesh cached)")
        tracking_mesh = scaled_mesh

    # -------------------------------------------------------------------------
    # Step 9: FoundationPose tracking
    # -------------------------------------------------------------------------
    if not _has_files(poses_dir):
        print("Step 9: FoundationPose tracking...")
        run_video_to_poses(
            video_path=video_path,
            depth_folder=tracking_depth_dir,
            masks_folder=f"{masks_dir}/{object_id}",
            camera_intrinsics_path=intrinsics_stable,
            mesh_path=tracking_mesh,
            poses_dir=poses_dir,
            weights_dir=fp_weights,
            reference_frame=reference_frame,
            reregister_iou_thresh=0.5,
            register_iteration=5,
            track_iteration=5,
            n_particles=128,
            particle_process_noise_t=0.005,
            particle_process_noise_r=0.02,
            particle_iteration=5,
        )
    else:
        print("Step 9: Skipping (poses already computed)")

    # -------------------------------------------------------------------------
    # Step 9b: FP-guided depth scale correction + re-track (optional)
    # -------------------------------------------------------------------------
    final_poses_dir = poses_dir
    if correct_depth:
        depth_corrected_dir = f"{output_dir}/depth_fp_corrected"
        poses_corrected_dir = f"{output_dir}/poses_corrected"
        if not _has_files(poses_corrected_dir):
            if not _has_files(depth_corrected_dir):
                print("Step 9b: Correcting depth scale via FP feedback...")
                run_correct_depth_scale(
                    poses_dir=poses_dir,
                    mesh_path=tracking_mesh,
                    depth_folder=tracking_depth_dir,
                    intrinsics_path=intrinsics_stable,
                    output_folder=depth_corrected_dir,
                    masks_folder=f"{masks_dir}/{object_id}",
                )
            print("Step 9b: Re-tracking with corrected depth...")
            run_video_to_poses(
                video_path=video_path,
                depth_folder=depth_corrected_dir,
                masks_folder=f"{masks_dir}/{object_id}",
                camera_intrinsics_path=intrinsics_stable,
                mesh_path=tracking_mesh,
                poses_dir=poses_corrected_dir,
                weights_dir=fp_weights,
                reference_frame=reference_frame,
                reregister_iou_thresh=0.3,
                register_iteration=5,
                track_iteration=5,
                n_particles=128,
                particle_process_noise_t=0.005,
                particle_process_noise_r=0.02,
                particle_iteration=10,
            )
        else:
            print("Step 9b: Skipping (corrected poses cached)")
        final_poses_dir = poses_corrected_dir
        tracking_depth_dir = depth_corrected_dir

    # -------------------------------------------------------------------------
    # Step 10: EKF + RTS pose smoothing
    # -------------------------------------------------------------------------
    if not _has_files(poses_smoothed_dir):
        print("Step 10: EKF smoothing poses...")
        run_ekf_smoothing(
            poses_dir=final_poses_dir,
            mesh_path=tracking_mesh,
            intrinsics_path=intrinsics_stable,
            weights_dir=fp_weights,
            output_dir=poses_smoothed_dir,
            masks_folder=f"{masks_dir}/{object_id}",
            process_noise_xy=0.02,
            process_noise_z=0.02,
            process_noise_r=0.02,
            measurement_noise_xy=0.02,
            measurement_noise_z=0.1,
            measurement_noise_r=0.05,
            min_iou=0.1,
        )
    else:
        print("Step 10: Skipping (smoothed poses cached)")

    # -------------------------------------------------------------------------
    # Step 11: Render mesh overlays
    # -------------------------------------------------------------------------
    if not _has_files(renders_dir):
        print("Step 11: Rendering mesh overlays...")
        run_render_poses(
            mesh_path=tracking_mesh,
            poses_dir=poses_smoothed_dir,
            frames_dir=frames_dir,
            intrinsics_path=intrinsics_stable,
            output_dir=renders_dir,
        )
    else:
        print("Step 11: Skipping (renders cached)")

    # -------------------------------------------------------------------------
    # Step 12: Encode + stitch videos
    # -------------------------------------------------------------------------
    print("Step 12: Encoding videos...")
    renders_mp4 = f"{output_dir}/renders.mp4"
    masks_mp4   = f"{output_dir}/masks.mp4"
    depth_mp4   = f"{output_dir}/depth.mp4"
    frames_to_video(renders_dir, renders_mp4)
    frames_to_video(f"{masks_dir}/{object_id}", masks_mp4)
    frames_to_video(tracking_depth_dir, depth_mp4)
    stitch_videos([renders_mp4, masks_mp4, depth_mp4], f"{output_dir}/comparison.mp4")

    print(f"Done. Comparison video: {output_dir}/comparison.mp4")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track a pre-made mesh against a video")
    parser.add_argument("--video_path",       required=True,  help="Input video file")
    parser.add_argument("--mesh_path",        required=True,  help="Path to existing mesh (GLB/OBJ)")
    parser.add_argument("--output_dir",       required=True,  help="Root output directory")
    parser.add_argument("--detection_prompt", required=True,  help="Text prompt for Grounding DINO")
    parser.add_argument("--depth_model",      default="moge", choices=["moge", "unidepth", "depth_anything"])
    parser.add_argument("--depth_weights",    required=True,  help="Path to depth model weights")
    parser.add_argument("--sam2_weights",     required=True,  help="Path to SAM2 weights")
    parser.add_argument("--fp_weights",       required=True,  help="Path to FoundationPose weights")
    parser.add_argument("--dino_weights",     required=True,  help="Path to Grounding DINO weights")
    parser.add_argument("--object_id",        type=int,   default=1)
    parser.add_argument("--reference_frame",  type=int,   default=40)
    parser.add_argument("--simplify_factor",  type=float, default=None, help="Mesh simplification factor (0–1)")
    parser.add_argument("--align_depth",      action="store_true", help="Align depth sequence to reference frame")
    parser.add_argument("--no_estimate_scale",action="store_true", help="Skip mesh scale estimation")
    parser.add_argument("--correct_depth",    action="store_true", help="FP-guided depth correction + re-track")
    args = parser.parse_args()

    run_premade_mesh_tracking(
        video_path=args.video_path,
        mesh_path=args.mesh_path,
        detection_prompt=args.detection_prompt,
        output_dir=args.output_dir,
        depth_model=args.depth_model,
        depth_weights=args.depth_weights,
        sam2_weights=args.sam2_weights,
        fp_weights=args.fp_weights,
        dino_weights=args.dino_weights,
        object_id=args.object_id,
        reference_frame=args.reference_frame,
        simplify_factor=args.simplify_factor,
        align_depth=args.align_depth,
        estimate_scale=not args.no_estimate_scale,
        correct_depth=args.correct_depth,
    )
