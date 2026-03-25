from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.detectron2.docker._config import IMAGE_NAME, MODULES_DIR

_LIB_CONFIG = Path(__file__).parent.parent / "lib" / "mv_config.yaml"


def run_mv_track_bboxes(
    weights_dir: str,
    output_dir: str,
    image_dir: str | None = None,
    video_dir: str | None = None,
    config_path: str = str(_LIB_CONFIG),
    debug: int = -1,
    dev: bool = False,
) -> None:
    inputs = {
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
        module="v2d.detectron2.lib.mv_track_bboxes",
        inputs=inputs,
        outputs=outputs,
        extra_args={"debug": debug if debug >= 0 else None},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={
            "PYTHONUNBUFFERED": "1",
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-view bbox tracking")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_dir", type=str, help="Directory containing images")
    input_group.add_argument("--video_dir", type=str, help="Directory containing videos")

    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing model weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for outputs")
    parser.add_argument("--config_path", type=str, default=str(_LIB_CONFIG), help="Path to mv_config.yaml")
    parser.add_argument("--debug", type=int, default=-1, help="Debug level")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_track_bboxes(
        weights_dir=args.weights_dir,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        video_dir=args.video_dir,
        config_path=args.config_path,
        debug=args.debug,
        dev=args.dev,
    )
