import os

from v2d.detectron2.docker.run_mv_track_bboxes import run_mv_track_bboxes
from v2d.sam3d_body.docker.run_mv_optimize_mhr_params import run_mv_optimize_mhr_params


RECON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
DATA_DIR = os.path.expanduser("~/datasets/v2d/rect_2026-03-12_16-58-57_tall_bar_stool_criss_cross_apple_sauce_01")


def main():
    # run_mv_track_bboxes(
    #     weights_dir=os.path.join(RECON_DIR, "data/weights/detectron2"),
    #     output_dir=os.path.join(DATA_DIR, "detectron2"),
    #     image_dir=os.path.join(DATA_DIR, "images"),
    #     dev=True,
    # )

    run_mv_optimize_mhr_params(
        camera_params_path=os.path.join(DATA_DIR, "edex"),
        weights_dir=os.path.join(RECON_DIR, "data/weights/sam3d_body"),
        bbox_dir=os.path.join(DATA_DIR, "detectron2"),
        output_dir=os.path.join(DATA_DIR, "sam3d_body"),
        image_dir=os.path.join(DATA_DIR, "images"),
        dev=True,
    )

if __name__ == "__main__":
    main()
