# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Pipeline: Video → Segment → Depth → 3D Mesh → Scale Align → Track → Render

Steps:
  1. Extract frames   — video → frames/
  2. SAM2             — video + prompts → per-object masks/
  3. MoGe             — video → depth/ + intrinsics/
  4. SAM3D            — frame[0] + mask[0] → mesh + transform + intrinsics
  5. Simplify mesh    — reduce polygon count for faster tracking
  6. Align depth      — simplified mesh + SAM3D transform + MoGe depth[0] → metric scale Transform3d
  7. Scale mesh       — apply scale to object-space mesh → scale-correct mesh
  8. Track poses      — FoundationPose: video + depth + masks + mesh → poses/
  9. Render overlays  — mesh + poses + frames → renders/

Run from reconstruction/:
    python -m v2d.pipelines.run_video_object_tracking
"""

from v2d.pipelines.extract_images import extract_images
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.moge.docker.run_video_to_depth import run_video_to_depth
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
from v2d.mesh.docker.run_mesh_simplify import run_mesh_simplify
from v2d.mesh.docker.run_mesh_transform import run_mesh_transform
from v2d.mesh.docker.run_mesh_align_depth import run_mesh_align_depth
from v2d.mesh.docker.run_mesh_render_image import run_mesh_render_image
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses


def run_video_object_tracking(
    video_path: str,
    prompts_path: str,
    object_id: int,
    output_dir: str,
    sam2_weights: str,
    moge_weights: str,
    sam3d_weights: str,
    fp_weights: str,
    reference_frame: int = 20,
    simplify_factor: float = 0.5,
) -> None:
    """
    Full E2E pipeline from video to tracked mesh overlays.

    Args:
        video_path:       Input video file.
        prompts_path:     SAM2 prompts JSON (bounding boxes / clicks per object).
        object_id:        Which object_id from the prompts to track.
        output_dir:       Root directory for all intermediate and final outputs.
        sam2_weights:     Path to SAM2 model weights directory.
        moge_weights:     Path to MoGe model weights directory.
        sam3d_weights:    Path to SAM3D model weights directory.
        fp_weights:       Path to FoundationPose model weights directory.
        reference_frame:  Frame index used for SAM3D reconstruction and FP registration.
        simplify_factor:  Target face fraction for mesh simplification (0.0–1.0).
    """

    # Intermediate paths
    frames_dir     = f"{output_dir}/frames"
    masks_dir      = f"{output_dir}/masks"
    depth_dir      = f"{output_dir}/depth"
    intrinsics_dir = f"{output_dir}/intrinsics"

    sam3d_mesh      = f"{output_dir}/sam3d_mesh.glb"
    sam3d_transform = f"{output_dir}/sam3d_transform.json"
    sam3d_intrinsics = f"{output_dir}/sam3d_intrinsics.json"

    simplified_mesh = f"{output_dir}/simplified_mesh.glb"
    scale_transform = f"{output_dir}/scale_transform.json"
    scaled_mesh     = f"{output_dir}/scaled_mesh.glb"

    poses_dir  = f"{output_dir}/poses"
    renders_dir = f"{output_dir}/renders"

    ref = f"{reference_frame:06d}"

    # -------------------------------------------------------------------------
    # Step 1: Extract frames
    #   Video → frames/{000000,000001,...}.png
    # -------------------------------------------------------------------------
    print("Step 1: Extracting frames...")
    extract_images(video_path, frames_dir)

    # # -------------------------------------------------------------------------
    # # Step 2: SAM2 segmentation
    # #   Video + prompts → masks/{object_id}/{000000,...}.png
    # # -------------------------------------------------------------------------
    print("Step 2: SAM2 segmentation...")
    run_video_to_masks(
        video_path=video_path,
        prompts_path=prompts_path,
        masks_dir=masks_dir,
        weights_dir=sam2_weights,
    )

    # # -------------------------------------------------------------------------
    # # Step 3: MoGe depth estimation
    # #   Video → depth/{000000,...}.png + intrinsics/{000000,...}.json
    # # -------------------------------------------------------------------------
    print("Step 3: MoGe depth estimation...")
    run_video_to_depth(
        video_path=video_path,
        depth_folder=depth_dir,
        intrinsics_folder=intrinsics_dir,
        weights_path=moge_weights,
    )

    # # -------------------------------------------------------------------------
    # # Step 4: SAM3D mesh reconstruction
    # #   Reference frame image + object mask → mesh + transform + intrinsics
    # # -------------------------------------------------------------------------
    print("Step 4: SAM3D mesh reconstruction...")
    run_image_to_mesh(
        image_path=f"{frames_dir}/{ref}.png",
        mask_path=f"{masks_dir}/{object_id}/{ref}.png",
        mesh_path=sam3d_mesh,
        transform_path=sam3d_transform,
        intrinsics_path=sam3d_intrinsics,
        weights_dir=sam3d_weights,
        with_layout_postprocess=True
    )

    # -------------------------------------------------------------------------
    # Step 5: Simplify mesh
    #   Reduce polygon count for faster FoundationPose tracking
    # -------------------------------------------------------------------------
    print("Step 5: Simplifying mesh...")
    run_mesh_simplify(
        input_mesh_path=sam3d_mesh,
        output_mesh_path=simplified_mesh,
        factor=simplify_factor,
    )

    # -------------------------------------------------------------------------
    # Step 6: Estimate metric scale
    #   Pose simplified_mesh via sam3d_transform for depth comparison, then find
    #   the scale factor s such that s * simplified_mesh → metric object-space.
    #   (run_mesh_align_depth composes the depth-correction with the transform
    #   scale so the output is directly applicable to simplified_mesh.)
    #   Output: scale-only Transform3d JSON
    # -------------------------------------------------------------------------
    print("Step 6: Aligning mesh scale to MoGe depth...")
    run_mesh_align_depth(
        mesh_path=simplified_mesh,
        depth_path=f"{depth_dir}/{ref}.png",
        intrinsics_path=f"{intrinsics_dir}/{ref}.json",
        output_transform_path=scale_transform,
        transform_path=sam3d_transform,
    )

    # -------------------------------------------------------------------------
    # Step 7: Apply scale to object-space mesh
    #   Scale the simplified mesh so its vertices are in real-world metric units
    #   consistent with MoGe depth.
    # -------------------------------------------------------------------------
    print("Step 7: Scaling mesh to real-world units...")
    run_mesh_transform(
        input_mesh_path=simplified_mesh,
        transform_path=scale_transform,
        output_mesh_path=scaled_mesh,
    )

    # -------------------------------------------------------------------------
    # Step 8: FoundationPose tracking
    #   Track the scale-correct mesh across all video frames.
    #   Uses MoGe depth + SAM2 object masks + reference-frame intrinsics.
    #   Output: poses/{000000,...}.json  (per-frame Transform3d)
    # -------------------------------------------------------------------------
    print("Step 8: FoundationPose tracking...")
    run_video_to_poses(
        video_path=video_path,
        depth_folder=depth_dir,
        masks_folder=f"{masks_dir}/{object_id}",
        camera_intrinsics_path=f"{intrinsics_dir}/{ref}.json",
        mesh_path=scaled_mesh,
        poses_dir=poses_dir,
        weights_dir=fp_weights,
        reference_frame=reference_frame,
    )

    # -------------------------------------------------------------------------
    # Step 9: Render mesh overlays
    #   For every frame: apply per-frame FP pose to scaled mesh, render with
    #   vertex colors, composite over the original video frame.
    #   Broadcast: 1 mesh × N poses × N frames × 1 intrinsics → N renders
    # -------------------------------------------------------------------------
    print("Step 9: Rendering mesh overlays...")
    run_mesh_render_image(
        mesh_path=scaled_mesh,
        intrinsics_path=f"{intrinsics_dir}/{ref}.json",
        output_image_path=f"{renders_dir}/*.png",
        transform_path=f"{poses_dir}/*.json",
        background_path=f"{frames_dir}/*.png",
    )

    print(f"Done. Renders written to {renders_dir}/")


def main():
    run_video_object_tracking(
        video_path="modules/v2d_sam2/assets/test_video.mp4",
        prompts_path="modules/v2d_sam2/assets/test_prompts.json",
        object_id=1,  # chair (object_id=1 in test_prompts.json)
        output_dir="data/outputs/object_tracking",
        sam2_weights="data/weights/sam2",
        moge_weights="data/weights/moge",
        sam3d_weights="data/weights/sam3d",
        fp_weights="data/weights/foundation_pose",
    )


if __name__ == "__main__":
    main()
