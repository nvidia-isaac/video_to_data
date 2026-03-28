import argparse
import os

from v2d.detectron2.docker.run_mv_track_bboxes import run_mv_track_bboxes
from v2d.sam2.docker.run_mv_videos_to_masks import run_mv_videos_to_masks
from v2d.sam3d_body.docker.run_mv_optimize_mhr_params import run_mv_optimize_mhr_params


RECON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def main(data_dir: str, dev: bool = False):
    # Track human bboxes
    run_mv_track_bboxes(
        weights_dir=os.path.join(RECON_DIR, "data/weights/detectron2"),
        output_dir=os.path.join(data_dir, "detectron2"),
        image_dir=os.path.join(data_dir, "images"),
        dev=dev,
    )

    # Get human masks from SAM2
    run_mv_videos_to_masks(
        weights_dir=os.path.join(RECON_DIR, "data/weights/sam2"),
        bbox_dir=os.path.join(data_dir, "detectron2"),
        output_dir=os.path.join(data_dir, "sam2"),
        image_dir=os.path.join(data_dir, "images"),
        dev=dev,
    )

    # Optimize MHR parameters from multiple views
    run_mv_optimize_mhr_params(
        camera_params_path=os.path.join(DATA_DIR, "edex"),
        weights_dir=os.path.join(RECON_DIR, "data/weights/sam3d_body"),
        bbox_dir=os.path.join(data_dir, "detectron2"),
        output_dir=os.path.join(data_dir, "sam3d_body"),
        image_dir=os.path.join(data_dir, "images"),
        dev=dev,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        dev=args.dev,
    )
