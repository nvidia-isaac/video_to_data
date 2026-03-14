"""
End-to-end test pipeline composing v2d Docker modules.
Run from reconstruction/ or repo root: python -m v2d.pipelines.run_test_pipeline

Steps:
  1. SAM2                    : video → masks
  2. MoGe                    : video → depth + intrinsics
  3. SAM3D                   : image → mesh + transform + intrinsics
  4. Simplify mesh
  5. Align mesh / estimate scale
  6. Transform mesh
  7. FoundationPose           : video → 6D object poses (+ debug frames)
  8. FoundationPose render    : poses + mesh → mesh overlay frames
  9. NLF                      : video → SMPL body pose
 10. NLF render               : SMPL params → body overlay frames
"""

from v2d.sam2.docker.run_video_to_masks import run_video_to_masks
from v2d.moge.docker.run_video_to_depth import run_video_to_depth
from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
from v2d.foundation_pose.docker.run_simplify_mesh import run_simplify_mesh
from v2d.foundation_pose.docker.run_align_mesh_scale import run_align_mesh_scale
from v2d.foundation_pose.docker.run_transform_mesh import run_transform_mesh
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.foundation_pose.docker.run_render_overlay import run_render_overlay
from v2d.nlf.docker.run_video_to_smpl import run_video_to_smpl
from v2d.nlf.docker.run_render_smpl_overlay import run_render_smpl_overlay


def main():
    # -------------------------------------------------------------------------
    # Step 1: SAM2 — segment objects in video to produce per-frame masks
    # -------------------------------------------------------------------------
    run_video_to_masks(
        video_path="modules/v2d_sam2/assets/test_video.mp4",
        prompts_path="modules/v2d_sam2/assets/prompts.json",
        masks_dir="data/outputs/sam2/masks",
        weights_dir="data/weights/sam2",
        dev=False,
    )

    # -------------------------------------------------------------------------
    # Step 2: MoGe — estimate per-frame depth and camera intrinsics from video
    # -------------------------------------------------------------------------
    run_video_to_depth(
        video_path="modules/v2d_sam2/assets/test_video.mp4",
        depth_folder="data/outputs/moge/depth",
        intrinsics_folder="data/outputs/moge/intrinsics",
        weights_path="data/weights/moge",
        dev=False,
    )

    # -------------------------------------------------------------------------
    # Step 3: SAM3D — reconstruct 3D mesh from the first video frame + object mask
    # -------------------------------------------------------------------------
    run_image_to_mesh(
        image_path="modules/v2d_moge/assets/test_image.jpg",  # first frame of test video
        mask_path="data/outputs/sam2/masks/1/000000.png",  # object mask
        mesh_path="data/outputs/sam3d/mesh_1.glb",
        transform_path="data/outputs/sam3d/transform_1.json",
        intrinsics_path="data/outputs/sam3d/intrinsics_1.json",
        weights_dir="data/weights/sam3d",
        dev=False,
    )

    # -------------------------------------------------------------------------
    # Step 4: Simplify mesh — reduce polygon count for faster downstream use
    # -------------------------------------------------------------------------
    run_simplify_mesh(
        input_mesh="data/outputs/sam3d/mesh_1.glb",
        output_mesh="data/outputs/sam3d/mesh_1_simplified.glb",
        factor=0.1,
        dev=False,
    )

    # -------------------------------------------------------------------------
    # Step 5: Align mesh / estimate scale — align mesh to depth map coordinate
    #         frame and recover metric scale using the first-frame depth + mask
    # -------------------------------------------------------------------------
    run_align_mesh_scale(
        mesh_path="data/outputs/sam3d/mesh_1_simplified.glb",
        depth_path="data/outputs/moge/depth/000000.png",
        mask_path="data/outputs/sam2/masks/1/000000.png",
        intrinsics_path="data/outputs/sam3d/intrinsics_1.json",
        transform_path="data/outputs/sam3d/transform_1.json",
        output_transform_path="data/outputs/sam3d/transform_1_aligned.json",
        dev=False,
    )

    # -------------------------------------------------------------------------
    # Step 6: Transform mesh — bake aligned transform into mesh geometry
    # -------------------------------------------------------------------------
    run_transform_mesh(
        input_mesh="data/outputs/sam3d/mesh_1_simplified.glb",
        output_mesh="data/outputs/sam3d/mesh_1_transformed.glb",
        transform_path="data/outputs/sam3d/transform_1_aligned.json",
        dev=False,
    )

    # -------------------------------------------------------------------------
    # Step 7: FoundationPose — track 6D object pose across all video frames
    # -------------------------------------------------------------------------
    run_video_to_poses(
        video_path="modules/v2d_sam2/assets/test_video.mp4",
        depth_folder="data/outputs/moge/depth",
        masks_folder="data/outputs/sam2/masks/1",  # object mask
        camera_intrinsics_path="data/outputs/moge/intrinsics/000000.json",
        mesh_path="data/outputs/sam3d/mesh_1_transformed.glb",
        poses_dir="data/outputs/foundation_pose/poses",
        weights_dir="data/weights/foundation_pose",
        reference_frame=0,  # initialize pose from first frame
        debug_dir="data/outputs/foundation_pose/debug",
        dev=False,
    )

    # -------------------------------------------------------------------------
    # Step 8: FoundationPose render — overlay mesh on video using tracked poses
    # -------------------------------------------------------------------------
    run_render_overlay(
        video_path="modules/v2d_sam2/assets/test_video.mp4",
        poses_dir="data/outputs/foundation_pose/poses",
        mesh_path="data/outputs/sam3d/mesh_1_transformed.glb",
        camera_intrinsics_path="data/outputs/moge/intrinsics/000000.json",
        output_dir="data/outputs/foundation_pose/overlay",
        dev=False,
    )

    # -------------------------------------------------------------------------
    # Step 9: NLF — estimate SMPL human body pose across all video frames
    # -------------------------------------------------------------------------
    run_video_to_smpl(
        video_path="modules/v2d_sam2/assets/test_video.mp4",
        masks_dir="data/outputs/sam2/masks/0",  # person mask
        intrinsics_path="data/outputs/moge/intrinsics/000000.json",
        gender="male",
        output_path="data/outputs/nlf/smpl_params.h5",
        weights_dir="data/weights/nlf",
        model_type="smplh",
        chunk_size=32,
        dev=False,
    )

    # -------------------------------------------------------------------------
    # Step 10: NLF render — overlay SMPL body mesh on video
    # -------------------------------------------------------------------------
    run_render_smpl_overlay(
        video_path="modules/v2d_sam2/assets/test_video.mp4",
        smpl_params_path="data/outputs/nlf/smpl_params.h5",
        intrinsics_path="data/outputs/moge/intrinsics/000000.json",
        output_dir="data/outputs/nlf/overlay",
        weights_dir="data/weights/nlf",
        dev=False,
    )


if __name__ == "__main__":
    main()
