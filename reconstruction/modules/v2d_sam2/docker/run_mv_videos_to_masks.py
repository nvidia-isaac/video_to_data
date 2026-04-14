from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.sam2.docker._config import IMAGE_NAME, MODULES_DIR

_LIB_CONFIG = Path(__file__).parent.parent / "lib" / "mv_videos_to_masks.yaml"


def run_mv_videos_to_masks(
    bbox_dir: str,
    output_dir: str,
    weights_dir: str,
    image_dir: str | None = None,
    video_dir: str | None = None,
    config_path: str = str(_LIB_CONFIG),
    dev: bool = False,
) -> None:
    inputs = {
        "bbox_dir": bbox_dir,
        "weights_dir": weights_dir,
        "config_path": config_path,
    }
    if image_dir:
        inputs["image_dir"] = image_dir
    if video_dir:
        inputs["video_dir"] = video_dir

    outputs = {"output_dir": output_dir}

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.sam2.lib.mv_videos_to_masks",
        inputs=inputs,
        outputs=outputs,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={"PYTHONUNBUFFERED": "1"},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-view video/image to masks using SAM2 with detectron2 bbox prompts"
    )
    parser.add_argument("--bbox_dir", type=str, required=True,
                        help="Directory containing per-camera bbox_track .pt files")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_dir", type=str, help="Directory containing images")
    input_group.add_argument("--video_dir", type=str, help="Directory containing videos")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for per-camera masks")
    parser.add_argument("--weights_dir", type=str, required=True,
                        help="Path to SAM2 weights directory")
    parser.add_argument("--config_path", type=str, default=str(_LIB_CONFIG),
                        help="Path to config YAML")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_videos_to_masks(
        bbox_dir=args.bbox_dir,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        image_dir=args.image_dir,
        video_dir=args.video_dir,
        config_path=args.config_path,
        dev=args.dev,
    )
