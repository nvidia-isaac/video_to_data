"""
Run CuSFM structure-from-motion on a directory of stereo images.

Inputs:
  --input_dir   Directory containing images and frames_meta.json
  --output_dir  Output directory for SfM results

Output:
  <output_dir>/keyframes/frames_meta.json — camera poses per keyframe
"""
import argparse
import os
from pathlib import Path

from v2d.docker.container import run_in_container

IMAGE_NAME = "v2d_cusfm"
_MODULES_DIR = str(Path(__file__).parents[3])  # reconstruction/modules/


def run_image_list_to_sfm(
    input_dir: str,
    output_dir: str,
    config_set: str = "backpack",
    feature_type: str = None,
    num_threads: int = 1,
    dev: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    extra = {"config_set": config_set, "num_threads": num_threads}
    if feature_type:
        extra["feature_type"] = feature_type
    run_in_container(
        image=IMAGE_NAME,
        module="v2d_cusfm.lib.image_list_to_sfm",
        inputs={"input_dir": input_dir},
        outputs={"output_dir": output_dir},
        extra_args=extra,
        dev=dev,
        modules_dir=_MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CuSFM on a directory of images")
    parser.add_argument("--input_dir",    required=True, help="Input images + frames_meta.json")
    parser.add_argument("--output_dir",   required=True, help="Output directory for SfM results")
    parser.add_argument("--config_set",   default="backpack", help="Config preset (backpack|av|isaac|rgbd)")
    parser.add_argument("--feature_type", default=None,   help="Feature type override")
    parser.add_argument("--num_threads",  type=int, default=1, help="CPU threads")
    parser.add_argument("--dev",          action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_image_list_to_sfm(
        args.input_dir,
        args.output_dir,
        config_set=args.config_set,
        feature_type=args.feature_type,
        num_threads=args.num_threads,
        dev=args.dev,
    )
