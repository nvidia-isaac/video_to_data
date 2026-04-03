"""Pipeline: rosbag → preprocessing → depth → detection → segmentation → body reconstruction.

Usage:
    python -m v2d.pipelines.run_mv_reconstruction \
        --rosbag_path /data/rosbags/2026-03-28_session1 \
        --data_dir /data/datasets/2026-03-28_session1 \
        --extrinsics_edex_path /data/datasets/2026-03-28_calibration/extrinsics/edex
"""

import argparse
import os

# from v2d.rosbag.docker.run_rosbag_to_edex import run_rosbag_to_edex
# from v2d.preprocess.docker.run_mv_preprocess_stereo import run_mv_preprocess_stereo
# from v2d.foundation_stereo.docker.run_mv_image_list_to_depth import run_mv_image_list_to_depth
from v2d.detectron2.docker.run_mv_track_bboxes import run_mv_track_bboxes
from v2d.sam2.docker.run_mv_videos_to_masks import run_mv_videos_to_masks
from v2d.sam3d_body.docker.run_mv_optimize_mhr_params import run_mv_optimize_mhr_params
from v2d.sam3d_body.docker.run_mv_eval_chamfer import run_mv_eval_chamfer

RECON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def main(
    rosbag_path: str,
    data_dir: str,
    extrinsics_edex_path: str,
    dev: bool = False,
):
    raw_dir = os.path.join(data_dir, "raw")

    # # 1. Extract images + intrinsics from rosbag
    # run_rosbag_to_edex(
    #     rosbag_path=rosbag_path,
    #     output_dir=raw_dir,
    #     no_extrinsics=True,
    #     dev=dev,
    # )

    # # 2. Stereo preprocessing (rectification, rescaling, video encoding)
    # run_mv_preprocess_stereo(
    #     image_dir=os.path.join(raw_dir, "images"),
    #     output_dir=data_dir,
    #     edex_path=os.path.join(raw_dir, "edex"),
    #     extrinsics_edex_path=extrinsics_edex_path,
    #     dev=dev,
    # )

    # # 3. Stereo depth estimation
    # run_mv_image_list_to_depth(
    #     camera_params_path=os.path.join(data_dir, "edex"),
    #     image_dir=os.path.join(data_dir, "images"),
    #     output_dir=os.path.join(data_dir, "foundation_stereo"),
    #     model_dir=os.path.join(RECON_DIR, "data/weights/foundation_stereo"),
    #     dev=dev,
    # )

    # 4. Detect + track human bounding boxes
    run_mv_track_bboxes(
        weights_dir=os.path.join(RECON_DIR, "data/weights/detectron2"),
        output_dir=os.path.join(data_dir, "detectron2"),
        image_dir=os.path.join(data_dir, "images"),
        dev=dev,
    )

    # 5. Segment human masks with SAM2
    run_mv_videos_to_masks(
        weights_dir=os.path.join(RECON_DIR, "data/weights/sam2"),
        bbox_dir=os.path.join(data_dir, "detectron2"),
        output_dir=os.path.join(data_dir, "sam2"),
        image_dir=os.path.join(data_dir, "images"),
        dev=dev,
    )

    # 6. Optimize MHR body parameters from multiple views
    run_mv_optimize_mhr_params(
        camera_params_path=os.path.join(data_dir, "edex"),
        weights_dir=os.path.join(RECON_DIR, "data/weights/sam3d_body"),
        bbox_dir=os.path.join(data_dir, "detectron2"),
        output_dir=os.path.join(data_dir, "sam3d_body"),
        image_dir=os.path.join(data_dir, "images"),
        mask_dir=os.path.join(data_dir, "sam2"),
        dev=dev,
    )

    # 7. Evaluate chamfer distance (mesh vs. depth point clouds)
    run_mv_eval_chamfer(
        camera_params_path=os.path.join(data_dir, "edex"),
        output_dir=os.path.join(data_dir, "sam3d_body"),
        depth_dir=os.path.join(data_dir, "foundation_stereo"),
        mask_dir=os.path.join(data_dir, "sam2"),
        dev=dev,
    )

    print("\n=== Multi-View Reconstruction Complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-view reconstruction pipeline")
    parser.add_argument("--rosbag_path", type=str, required=True,
                        help="Path to the ROS bag")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root output directory for all pipeline outputs")
    parser.add_argument("--extrinsics_edex_path", type=str, required=True,
                        help="Path to calibration EDEX with extrinsics")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    main(
        rosbag_path=args.rosbag_path,
        data_dir=args.data_dir,
        extrinsics_edex_path=args.extrinsics_edex_path,
        dev=args.dev,
    )
