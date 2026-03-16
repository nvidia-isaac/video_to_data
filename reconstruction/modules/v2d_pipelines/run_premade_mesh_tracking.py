"""
Pipeline: Pre-made Mesh + Video → Track → Render

Skips 3D reconstruction — use this when you already have a mesh for the object.
The mesh is assumed to be in metric scale (e.g. from BundleSDF).

Steps:
  1. Extract frames   — video → frames/
  2. Grounding DINO   — reference frame + text prompt → bounding box detection
  3. SAM2             — video + auto-generated prompt → per-object masks/
  4. MoGe             — video → depth/ + intrinsics/
  5. Simplify mesh    — reduce polygon count for faster tracking (optional)
  6. Track poses      — FoundationPose: video + depth + masks + mesh → poses/
  7. Render overlays  — mesh + poses + frames → renders/
  8. Encode + stitch  — renders/ + masks/ + depth/ → renders.mp4 + comparison.mp4

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
from v2d.mesh.docker.run_mesh_render_image import run_mesh_render_image
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses

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
    """

    frames_dir     = f"{output_dir}/frames"
    masks_dir      = f"{output_dir}/masks"
    depth_dir      = f"{output_dir}/depth"
    intrinsics_dir = f"{output_dir}/intrinsics"
    poses_dir      = f"{output_dir}/poses"
    renders_dir    = f"{output_dir}/renders"

    dino_detections = f"{output_dir}/dino_detections.json"
    sam2_prompts    = f"{output_dir}/sam2_prompts.json"

    ref = f"{reference_frame:06d}"

    # -------------------------------------------------------------------------
    # Step 1: Extract frames
    #   Video → frames/{000000,000001,...}.png
    # -------------------------------------------------------------------------
    print("Step 1: Extracting frames...")
    extract_images(video_path, frames_dir)

    # # -------------------------------------------------------------------------
    # # Step 2: Grounding DINO detection on reference frame
    # #   reference frame image + text prompt → bounding box detections JSON
    # # -------------------------------------------------------------------------
    print("Step 2: Grounding DINO detection...")
    run_image_to_object_bboxes(
        image_path=f"{frames_dir}/{ref}.png",
        output_path=dino_detections,
        prompt=detection_prompt,
        model_dir=dino_weights,
    )
    prompts = _dino_detections_to_sam2_prompts(dino_detections, reference_frame, object_id)
    os.makedirs(output_dir, exist_ok=True)
    with open(sam2_prompts, "w") as f:
        json.dump(prompts.to_dict(), f, indent=2)
    print(f"  Top detection box written to {sam2_prompts}")

    # # -------------------------------------------------------------------------
    # # Step 3: SAM2 segmentation
    # #   Video + auto-generated prompt → masks/{object_id}/{000000,...}.png
    # # -------------------------------------------------------------------------
    print("Step 3: SAM2 segmentation...")
    run_video_to_masks(
        video_path=video_path,
        prompts_path=sam2_prompts,
        masks_dir=masks_dir,
        weights_dir=sam2_weights,
    )

    # # -------------------------------------------------------------------------
    # # Step 4: MoGe depth estimation
    # #   Video → depth/{000000,...}.png + intrinsics/{000000,...}.json
    # # -------------------------------------------------------------------------
    # print("Step 4: MoGe depth estimation...")
    run_video_to_depth(
        video_path=video_path,
        depth_folder=depth_dir,
        intrinsics_folder=intrinsics_dir,
        weights_path=moge_weights,
    )

    # -------------------------------------------------------------------------
    # Step 5: Simplify mesh (optional)
    #   Reduce polygon count for faster FoundationPose tracking.
    # # -------------------------------------------------------------------------
    tracking_mesh = mesh_path
    if simplify_factor is not None:
        simplified_mesh = f"{output_dir}/simplified_mesh.glb"
        print("Step 5: Simplifying mesh...")
        run_mesh_simplify(
            input_mesh_path=mesh_path,
            output_mesh_path=simplified_mesh,
            factor=simplify_factor,
        )
        tracking_mesh = simplified_mesh
    else:
        print("Step 5: Skipping mesh simplification.")

    # # -------------------------------------------------------------------------
    # # Step 6: FoundationPose tracking
    # #   Track the mesh across all video frames using MoGe depth + SAM2 masks.
    # #   Output: poses/{000000,...}.json  (per-frame Transform3d object-to-camera)
    # # -------------------------------------------------------------------------
    # print("Step 6: FoundationPose tracking...")
    run_video_to_poses(
        video_path=video_path,
        depth_folder=depth_dir,
        masks_folder=f"{masks_dir}/{object_id}",
        camera_intrinsics_path=f"{intrinsics_dir}/{ref}.json",
        mesh_path=tracking_mesh,
        poses_dir=poses_dir,
        weights_dir=fp_weights,
        reference_frame=reference_frame,
    )

    # # -------------------------------------------------------------------------
    # # Step 7: Render mesh overlays
    # #   For every frame: apply per-frame FP pose to mesh, render with vertex
    # #   colors, composite over the original video frame.
    # #   Broadcast: 1 mesh × N poses × N frames × 1 intrinsics → N renders
    # # -------------------------------------------------------------------------
    # print("Step 7: Rendering mesh overlays...")
    run_mesh_render_image(
        mesh_path=tracking_mesh,
        intrinsics_path=f"{intrinsics_dir}/{ref}.json",
        output_image_path=f"{renders_dir}/*.png",
        transform_path=f"{poses_dir}/*.json",
        background_path=f"{frames_dir}/*.png",
    )

    # -------------------------------------------------------------------------
    # Step 8: Encode frame folders to videos and stitch side-by-side
    # -------------------------------------------------------------------------
    print("Step 8: Encoding videos...")
    renders_mp4 = f"{output_dir}/renders.mp4"
    masks_mp4   = f"{output_dir}/masks.mp4"
    depth_mp4   = f"{output_dir}/depth.mp4"

    frames_to_video(renders_dir, renders_mp4)
    frames_to_video(f"{masks_dir}/{object_id}", masks_mp4)
    frames_to_video(depth_dir, depth_mp4)

    print("Step 8: Stitching videos side-by-side...")
    stitch_videos([renders_mp4, masks_mp4, depth_mp4], f"{output_dir}/comparison.mp4")

    print(f"Done. Comparison video written to {output_dir}/comparison.mp4")


def main():
    run_premade_mesh_tracking(
        video_path="data/objects/yellow_spray/sessions/Session_20260310_161346/Session_20260310_161346_color.mp4",
        mesh_path="data/objects/yellow_spray/meshes/bundlesdf/textured_mesh.obj",
        detection_prompt="yellow spray can",
        object_id=1,
        output_dir="data/outputs/Session_20260310_161346_bundlesdf",
        sam2_weights="data/weights/sam2",
        moge_weights="data/weights/moge",
        fp_weights="data/weights/foundation_pose",
        dino_weights="data/weights/grounding_dino",
    )


if __name__ == "__main__":
    main()
