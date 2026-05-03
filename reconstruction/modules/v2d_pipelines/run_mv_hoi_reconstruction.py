"""Pipeline: rosbag -> preprocessing -> depth -> detection -> segmentation -> body reconstruction.

Usage:
    python -m v2d.pipelines.run_mv_hoi_reconstruction \
        --rosbag_path /data/rosbags/2026-03-28_session1 \
        --output_dir /data/datasets/2026-03-28_session1 \
        --extrinsics_camera_params_path /data/datasets/2026-03-28_calibration/extrinsics/edex
"""

import argparse
import os

from v2d.rosbag.docker.run_rosbag_to_edex import run_rosbag_to_edex
from v2d.mv.preprocess.docker.run_mv_preprocess import run_mv_preprocess
from v2d.foundation_stereo.docker.run_mv_image_list_to_depth import run_mv_image_list_to_depth
from v2d.grounding_dino.docker.run_mv_image_list_to_object_bboxes import run_mv_image_list_to_object_bboxes
from v2d.detectron2.docker.run_mv_track_bboxes import run_mv_track_bboxes
from v2d.sam2.docker.run_mv_videos_to_masks import run_mv_videos_to_masks
from v2d.foundation_pose.docker.run_mv_videos_to_poses import run_mv_videos_to_poses
from v2d.sam3d_body.docker.run_mv_optimize_mhr_params import run_mv_optimize_mhr_params
from v2d.sam3d_body.docker.run_export_soma import run_export_soma
from v2d.mv.postprocess.docker.run_mv_estimate_ground_plane import run_mv_estimate_ground_plane
from v2d.mv.postprocess.docker.run_mv_export_fused_pointcloud import run_mv_export_fused_pointcloud
from v2d.mv.postprocess.docker.run_mv_eval_chamfer_human import run_mv_eval_chamfer_human
from v2d.mv.postprocess.docker.run_mv_eval_chamfer_object import run_mv_eval_chamfer_object
from v2d.mv.postprocess.docker.run_mv_render_hoi_overlay import run_mv_render_hoi_overlay
from v2d.mv.postprocess.docker.run_mv_visualize_wis3d import run_mv_visualize_wis3d

RECON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MV_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mv_configs")


def _find_pinned_mesh(mesh_dir: str) -> str:
    """Locate the symmetry-aligned object mesh in a preprocess object_mesh dir."""
    path = os.path.join(mesh_dir, "output_aligned.glb")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"output_aligned.glb not found in {mesh_dir}")
    return path


def main(
    rosbag_path: str,
    output_dir: str,
    extrinsics_camera_params_path: str,
    obj_mesh_path: str,
    dev: bool = False,
):
    raw_dir = os.path.join(output_dir, "raw")
    preprocess_dir = os.path.join(output_dir, "preprocess")
    preprocess_images_dir = os.path.join(preprocess_dir, "images")
    preprocess_mesh_dir = os.path.join(preprocess_dir, "object_mesh")
    foundation_stereo_dir = os.path.join(output_dir, "foundation_stereo")
    grounding_dino_dir = os.path.join(output_dir, "grounding_dino")
    sam2_object_dir = os.path.join(output_dir, "sam2", "object")
    foundation_pose_dir = os.path.join(output_dir, "foundation_pose")
    detectron2_dir = os.path.join(output_dir, "detectron2")
    sam2_human_dir = os.path.join(output_dir, "sam2", "human")
    sam3d_body_dir = os.path.join(output_dir, "sam3d_body")
    export_soma_dir = os.path.join(output_dir, "sam3d_body", "export_soma")
    ground_plane_dir = os.path.join(output_dir, "postprocess", "ground_plane")
    fused_pointcloud_dir = os.path.join(output_dir, "postprocess", "fused_pointcloud")
    chamfer_human_dir = os.path.join(output_dir, "postprocess", "chamfer_human")
    chamfer_object_dir = os.path.join(output_dir, "postprocess", "chamfer_object")
    hoi_overlay_dir = os.path.join(output_dir, "postprocess", "hoi_overlay")
    wis3d_dir = os.path.join(output_dir, "postprocess", "wis3d")

    # # Extract images + intrinsics from rosbag
    # run_rosbag_to_edex(
    #     rosbag_path=rosbag_path,
    #     output_dir=raw_dir,
    #     no_extrinsics=True,
    #     dev=dev,
    # )

    # Preprocessing (rectification, rescaling, video encoding, HOI bbox remap)
    run_mv_preprocess(
        rgb_dir=os.path.join(raw_dir, "images"),
        output_dir=preprocess_dir,
        camera_params_path=os.path.join(raw_dir, "edex"),
        extrinsics_camera_params_path=extrinsics_camera_params_path,
        hoi_metadata_path=os.path.join(rosbag_path, "hoi_metadata.yaml"),
        mesh_path=obj_mesh_path,
        dev=dev,
    )

    # # Stereo depth estimation
    # run_mv_image_list_to_depth(
    #     camera_params_path=os.path.join(preprocess_dir, "edex"),
    #     rgb_dir=preprocess_images_dir,
    #     output_dir=foundation_stereo_dir,
    #     model_dir=os.path.join(RECON_DIR, "data/weights/foundation_stereo"),
    #     dev=dev,
    # )

    # Detect object bounding boxes with Grounding DINO
    run_mv_image_list_to_object_bboxes(
        rgb_dir=preprocess_images_dir,
        prompt_path=os.path.join(preprocess_dir, "prompt.txt"),
        output_dir=grounding_dino_dir,
        model_dir=os.path.join(RECON_DIR, "data/weights/grounding_dino"),
        dev=dev,
    )

    # Segment object masks with SAM2 (using grounding dino bboxes)
    run_mv_videos_to_masks(
        weights_dir=os.path.join(RECON_DIR, "data/weights/sam2"),
        bbox_dir=grounding_dino_dir,
        output_dir=sam2_object_dir,
        rgb_dir=preprocess_images_dir,
        config_path=os.path.join(MV_CONFIGS_DIR, "mv_videos_to_object_masks.yaml"),
        dev=dev,
    )

    # # Track object pose with FoundationPose (requires depth + object masks)
    # sym_json = os.path.join(preprocess_mesh_dir, "output_symmetry.json")
    # run_mv_videos_to_poses(
    #     camera_params_path=os.path.join(preprocess_dir, "edex"),
    #     rgb_dir=preprocess_images_dir,
    #     depth_dir=foundation_stereo_dir,
    #     mask_dir=sam2_object_dir,
    #     mesh_path=_find_pinned_mesh(preprocess_mesh_dir),
    #     symmetry_path=sym_json if os.path.exists(sym_json) else None,
    #     weights_dir=os.path.join(RECON_DIR, "data/weights/foundation_pose"),
    #     output_dir=foundation_pose_dir,
    #     dev=dev,
    # )

    # Detect + track human bounding boxes
    run_mv_track_bboxes(
        weights_dir=os.path.join(RECON_DIR, "data/weights/detectron2"),
        output_dir=detectron2_dir,
        rgb_dir=preprocess_images_dir,
        dev=dev,
    )

    # Segment human masks with SAM2
    run_mv_videos_to_masks(
        weights_dir=os.path.join(RECON_DIR, "data/weights/sam2"),
        bbox_dir=detectron2_dir,
        output_dir=sam2_human_dir,
        rgb_dir=preprocess_images_dir,
        dev=dev,
    )

    # Optimize MHR body parameters from multiple views
    run_mv_optimize_mhr_params(
        camera_params_path=os.path.join(preprocess_dir, "edex"),
        weights_dir=os.path.join(RECON_DIR, "data/weights/sam3d_body"),
        bbox_dir=detectron2_dir,
        output_dir=sam3d_body_dir,
        rgb_dir=preprocess_images_dir,
        mask_dir=sam2_human_dir,
        dev=dev,
    )

    # # Export MHR parameters to SOMA format
    # run_export_soma(
    #     params_path=os.path.join(sam3d_body_dir, "mhr_params_mv.pt"),
    #     output_path=os.path.join(export_soma_dir, "soma_params.npz"),
    #     mesh_path=os.path.join(sam3d_body_dir, "mhr_mesh_mv.pt"),
    #     weights_dir=os.path.join(RECON_DIR, "data/weights/sam3d_body"),
    #     autograd_iters=100,
    #     foot_weight=50.0,
    #     debug=1,
    #     dev=dev,
    # )

    # # Export fused multiview point clouds
    # run_mv_export_fused_pointcloud(
    #     camera_params_path=os.path.join(preprocess_dir, "edex"),
    #     depth_dir=foundation_stereo_dir,
    #     rgb_dir=preprocess_images_dir,
    #     output_dir=fused_pointcloud_dir,
    #     dev=dev,
    # )

    # # Estimate ground plane from depth + MHR foot keypoints
    # run_mv_estimate_ground_plane(
    #     camera_params_path=os.path.join(preprocess_dir, "edex"),
    #     depth_dir=foundation_stereo_dir,
    #     human_pose_dir=sam3d_body_dir,
    #     output_dir=ground_plane_dir,
    #     rgb_dir=preprocess_images_dir,
    #     dev=dev,
    # )

    # # Evaluate chamfer distance for human mesh
    # run_mv_eval_chamfer_human(
    #     camera_params_path=os.path.join(preprocess_dir, "edex"),
    #     human_pose_dir=sam3d_body_dir,
    #     output_dir=chamfer_human_dir,
    #     depth_dir=foundation_stereo_dir,
    #     mask_dir=sam2_human_dir,
    #     dev=dev,
    # )

    # # Evaluate chamfer distance for object mesh
    # run_mv_eval_chamfer_object(
    #     camera_params_path=os.path.join(preprocess_dir, "edex"),
    #     object_mesh_path=_find_pinned_mesh(preprocess_mesh_dir),
    #     object_pose_dir=foundation_pose_dir,
    #     output_dir=chamfer_object_dir,
    #     depth_dir=foundation_stereo_dir,
    #     mask_dir=sam2_object_dir,
    #     dev=dev,
    # )

    # # Render HOI overlay videos (object + human mesh on camera frames)
    # run_mv_render_hoi_overlay(
    #     camera_params_path=os.path.join(preprocess_dir, "edex"),
    #     object_mesh_path=_find_pinned_mesh(preprocess_mesh_dir),
    #     object_pose_dir=foundation_pose_dir,
    #     human_pose_dir=sam3d_body_dir,
    #     output_dir=hoi_overlay_dir,
    #     rgb_dir=preprocess_images_dir,
    #     dev=dev,
    # )

    # # Generate Wis3D interactive 3D visualization
    # run_mv_visualize_wis3d(
    #     camera_params_path=os.path.join(preprocess_dir, "edex"),
    #     object_mesh_path=_find_pinned_mesh(preprocess_mesh_dir),
    #     object_pose_dir=foundation_pose_dir,
    #     human_pose_dir=sam3d_body_dir,
    #     output_dir=wis3d_dir,
    #     ground_plane_dir=ground_plane_dir,
    #     dev=dev,
    # )

    print("\n=== Multi-View Reconstruction Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-view reconstruction pipeline")
    parser.add_argument("--rosbag_path", type=str, required=True,
                        help="Path to the ROS bag")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory for all pipeline outputs")
    parser.add_argument("--extrinsics_camera_params_path", type=str, required=True,
                        help="Path to calibration camera params file with extrinsics")
    parser.add_argument("--obj_mesh_path", type=str, required=True,
                        help="Path to object mesh file (for FoundationPose tracking)")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    main(
        rosbag_path=args.rosbag_path,
        output_dir=args.output_dir,
        extrinsics_camera_params_path=args.extrinsics_camera_params_path,
        obj_mesh_path=args.obj_mesh_path,
        dev=args.dev,
    )
