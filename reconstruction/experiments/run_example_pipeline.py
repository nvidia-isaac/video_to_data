"""
Example pipeline composing v2d Docker modules.
Run from reconstruction/ or repo root: python experiments/run_example_pipeline.py
"""
from v2d.moge.docker.run_video_to_depth import run_video_to_depth


def main():
    run_video_to_depth(
        video_path="modules/v2d_moge/assets/test_video.mp4",
        depth_folder="data/outputs/moge/depth",
        intrinsics_folder="data/outputs/moge/intrinsics",
        weights_path="data/weights/moge",
        dev=False,
    )


if __name__ == "__main__":
    main()
