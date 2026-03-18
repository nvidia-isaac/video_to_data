"""
Pipeline: Pre-made Mesh + Video → Track → Render

Skips 3D reconstruction — use this when you already have a mesh for the object.
The mesh is assumed to be in metric scale (e.g. from BundleSDF).

Steps:
  1. Extract frames   — video → frames/
  2. Grounding DINO   — reference frame + text prompt → bounding box detection
  3. SAM2             — video + auto-generated prompt → per-object masks/
  4. MoGe             — video → depth/ + intrinsics/
  5. Align depth      — per-frame scale correction to reference frame via feature matching
  6. Simplify mesh    — reduce polygon count for faster tracking (optional)
  7. Estimate scale   — coarse-to-fine grid search to align mesh scale to MoGe depth
  8. Track poses      — FoundationPose: video + depth + masks + mesh → poses/
  9. Render overlays  — mesh + poses + frames → renders/ (GPU-batched nvdiffrast)
 10. Encode + stitch  — renders/ + masks/ + depth/ → renders.mp4 + comparison.mp4

Run from reconstruction/:
    python -m v2d.pipelines.run_premade_mesh_tracking
"""

import json
import os

from v2d.pipelines.extract_images import extract_images
from v2d.pipelines.frames_to_video import frames_to_video
from v2d.pipelines.stitch_videos import stitch_videos
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.moge.docker.run_video_to_depth import run_video_to_depth
from v2d.mesh.docker.run_mesh_simplify import run_mesh_simplify
from v2d.foundation_pose.docker.run_render_poses import run_render_poses
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.foundation_pose.docker.run_estimate_mesh_scale import run_estimate_mesh_scale
from v2d.foundation_pose.docker.run_ekf_smoothing import run_ekf_smoothing
from v2d.foundation_pose.docker.run_correct_depth_scale import run_correct_depth_scale
from v2d.depth.lib.align_depth_sequence import align_depth_sequence
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.mesh.docker.run_mesh_align_depth import run_mesh_align_depth
from v2d.mesh.docker.run_mesh_transform import run_mesh_transform
from v2d.common.datatypes import DepthImage, CameraIntrinsics

def _dino_detections_to_sam2_prompts(
    detections_path: str,
    frame_index: int,
    object_id: int,
) -> Sam2Prompts:
    """Convert the top Grounding DINO detection into a SAM2 prompt."""
    with open(detections_path) as f:
        detections = json.load(f)
    if not detections:
        raise RuntimeError(f"Grounding DINO found no detections in {detections_path}")
    top = detections[0]  # already sorted by confidence descending
    box = BoundingBox.from_dict(top["box"])
    return Sam2Prompts(prompts=[
        Sam2Prompt(frame_index=frame_index, object_id=object_id, box=box)
    ])


def run_premade_mesh_tracking(
    video_path: str,
    mesh_path: str,
    detection_prompt: str,
    object_id: int,
    output_dir: str,
    sam2_weights: str,
    moge_weights: str,
    fp_weights: str,
    dino_weights: str,
    reference_frame: int = 40,
    simplify_factor: float | None = None,
    align_depth: bool = False,
    estimate_scale: bool = True,
    correct_depth: bool = False,
) -> None:
    """
    Track a pre-made mesh against a video and render per-frame mesh overlays.

    Args:
        video_path:        Input video file.
        mesh_path:         Path to an existing mesh (GLB, OBJ, etc.).
        detection_prompt:  Text prompt for Grounding DINO (e.g. "yellow spray can").
        object_id:         Object ID to assign in SAM2 tracking.
        output_dir:        Root directory for all intermediate and final outputs.
        sam2_weights:      Path to SAM2 model weights directory.
        moge_weights:      Path to MoGe model weights directory.
        fp_weights:        Path to FoundationPose model weights directory.
        dino_weights:      Path to Grounding DINO model weights directory.
        reference_frame:   Frame index used for DINO detection and FP registration.
        simplify_factor:   Target face fraction for mesh simplification (0.0–1.0),
                           or None to skip simplification and use the mesh as-is.
        align_depth:       If True (default), correct per-frame depth scale drift by
                           aligning each frame to the reference via sparse feature matching.
        estimate_scale:    If True (default), run coarse-to-fine scale estimation to
                           align the mesh scale to the MoGe metric depth before tracking.
        correct_depth:     If True, after the first FP tracking pass compute a per-frame
                           depth scale correction (rendered FP depth vs MoGe depth at the
                           object region) and re-run tracking with corrected depth.
                           Helps when MoGe underestimates scale for close-up objects.
                           Default False.
    """

    frames_dir       = f"{output_dir}/frames"
    masks_dir        = f"{output_dir}/masks"
    depth_dir        = f"{output_dir}/depth"
    depth_aligned_dir = f"{output_dir}/depth_aligned"
    intrinsics_dir   = f"{output_dir}/intrinsics"
    poses_dir             = f"{output_dir}/poses"
    depth_fp_corrected_dir = f"{output_dir}/depth_fp_corrected"
    poses_corrected_dir   = f"{output_dir}/poses_corrected"
    renders_dir           = f"{output_dir}/renders"

    intrinsics_stable = f"{output_dir}/intrinsics_stable.json"
    intrinsics_input = f"{output_dir}/intrinsics_vipe.json"

    dino_detections = f"{output_dir}/dino_detections.json"
    sam2_prompts    = f"{output_dir}/sam2_prompts.json"

    ref = f"{reference_frame:06d}"

    # -------------------------------------------------------------------------
    # Step 1: Extract frames
    #   Video → frames/{000000,000001,...}.png
    # -------------------------------------------------------------------------
    # print("Step 1: Extracting frames...")
    # extract_images(video_path, frames_dir)

    # # # # -------------------------------------------------------------------------
    # # # # Step 2: Grounding DINO detection on reference frame
    # # # #   reference frame image + text prompt → bounding box detections JSON
    # # # # -------------------------------------------------------------------------
    # print("Step 2: Grounding DINO detection...")
    # run_image_to_object_bboxes(
    #     image_path=f"{frames_dir}/{ref}.png",
    #     output_path=dino_detections,
    #     prompt=detection_prompt,
    #     model_dir=dino_weights,
    # )
    # prompts = _dino_detections_to_sam2_prompts(dino_detections, reference_frame, object_id)
    # os.makedirs(output_dir, exist_ok=True)
    # with open(sam2_prompts, "w") as f:
    #     json.dump(prompts.to_dict(), f, indent=2)
    # print(f"  Top detection box written to {sam2_prompts}")

    # # # # -------------------------------------------------------------------------
    # # # # Step 3: SAM2 segmentation
    # # # #   Video + auto-generated prompt → masks/{object_id}/{000000,...}.png
    # # # # -------------------------------------------------------------------------
    # print("Step 3: SAM2 segmentation...")
    # run_video_to_masks(
    #     video_path=video_path,
    #     prompts_path=sam2_prompts,
    #     masks_dir=masks_dir,
    #     weights_dir=sam2_weights,
    # )

    # # # # -------------------------------------------------------------------------
    # # # # Step 4: MoGe depth estimation
    # # # #   Video → depth/{000000,...}.png + intrinsics/{000000,...}.json
    # # # # -------------------------------------------------------------------------
    # print("Step 4: MoGe depth estimation...")
    # run_video_to_depth(
    #     video_path=video_path,
    #     depth_folder=depth_dir,
    #     intrinsics_folder=intrinsics_dir,
    #     weights_path=moge_weights,
    #     input_intrinsics_path=intrinsics_input,
    # )

    intrinsics_stable = intrinsics_input
    # -------------------------------------------------------------------------
    # Step 4b: Stabilise intrinsics
    #   Median of per-frame fx/fy/cx/cy → single stable intrinsics JSON.
    #   Avoids per-frame focal-length jitter causing apparent scale changes in FP.
    # -------------------------------------------------------------------------
    # print("Step 4b: Stabilising intrinsics...")
    stabilize_intrinsics(
        intrinsics_folder=intrinsics_dir,
        output_path=intrinsics_stable,
    )

    tracking_depth_dir = depth_dir
    # -------------------------------------------------------------------------
    # Step 7: FoundationPose tracking
    #   Track the mesh across all video frames using MoGe depth + SAM2 masks.
    #   Output: poses/{000000,...}.json  (per-frame Transform3d object-to-camera)
    # -------------------------------------------------------------------------
    tracking_mesh = mesh_path
    print("Step 7: FoundationPose tracking...")
    run_video_to_poses(
        video_path=video_path,
        depth_folder=tracking_depth_dir,
        masks_folder=f"{masks_dir}/{object_id}",
        camera_intrinsics_path=intrinsics_stable,
        mesh_path=tracking_mesh,
        poses_dir=poses_dir,
        weights_dir=fp_weights,
        reference_frame=reference_frame,
        reregister_iou_thresh=0.3,
        register_iteration=5,
        track_iteration=5,
        n_particles=128,
        particle_process_noise_t = 0.005,
        particle_process_noise_r = 0.02,
        particle_iteration = 10
    )
    # -------------------------------------------------------------------------
    # Step 7b+7c: FP-guided depth scale correction (optional)
    #   Render mesh at each FP pose → compare rendered vs MoGe depth at object
    #   region → per-frame scale correction → re-track with corrected depth.
    # -------------------------------------------------------------------------
    final_poses_dir = poses_dir
    # -------------------------------------------------------------------------
    # Step 8: EKF + RTS pose smoothing
    #   Forward ESKF + RTS backward smoother on the raw FP poses.
    #   IoU-weighted measurement noise discounts frames where FP lost track.
    # -------------------------------------------------------------------------
    print("Step 8: EKF smoothing poses...")
    poses_smoothed_dir = f"{output_dir}/poses_smoothed"
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
    # poses_smoothed_dir = final_poses_dir

    # -------------------------------------------------------------------------
    # Step 9: Render mesh overlays
    #   GPU-batched nvdiffrast renderer: all poses rasterised in parallel.
    #   Much faster than pyrender (no per-frame mesh re-upload, CUDA batching).
    # -------------------------------------------------------------------------
    print("Step 9: Rendering mesh overlays...")
    run_render_poses(
        mesh_path=tracking_mesh,
        poses_dir=poses_smoothed_dir,
        frames_dir=frames_dir,
        intrinsics_path=intrinsics_stable,
        output_dir=renders_dir,
    )

    # -------------------------------------------------------------------------
    # Step 9: Encode frame folders to videos and stitch side-by-side
    # -------------------------------------------------------------------------
    print("Step 9: Encoding videos...")
    renders_mp4 = f"{output_dir}/renders.mp4"
    masks_mp4   = f"{output_dir}/masks.mp4"
    depth_mp4   = f"{output_dir}/depth.mp4"

    frames_to_video(renders_dir, renders_mp4)
    frames_to_video(f"{masks_dir}/{object_id}", masks_mp4)
    frames_to_video(tracking_depth_dir, depth_mp4)

    print("Step 9: Stitching videos side-by-side...")
    stitch_videos([renders_mp4, masks_mp4, depth_mp4], f"{output_dir}/comparison.mp4")

    print(f"Done. Comparison video written to {output_dir}/comparison.mp4")


def main():
    sessions_dir = "data/objects/airplane/sessions"
    sessions = sorted(
        d for d in os.listdir(sessions_dir)
        if os.path.isdir(os.path.join(sessions_dir, d))
    )
    sessions = [s for s in sessions if "Session_20260310_132206" in s]
    for session in sessions:
        print(f"\n{'='*60}\nProcessing {session}\n{'='*60}")
        run_premade_mesh_tracking(
            video_path=f"{sessions_dir}/{session}/{session}_color.mp4",
            mesh_path="data/objects/airplane/mesh_bundlesdf/textured_mesh.obj",
            detection_prompt="toy airplane",
            object_id=1,
            output_dir=f"data/outputs/airplane_{session}",
            sam2_weights="data/weights/sam2",
            moge_weights="data/weights/moge",
            fp_weights="data/weights/foundation_pose",
            dino_weights="data/weights/grounding_dino",
            correct_depth=False,
        )


if __name__ == "__main__":
    main()
