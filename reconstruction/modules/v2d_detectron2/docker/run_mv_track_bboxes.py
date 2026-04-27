from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.detectron2.docker._config import IMAGE_NAME, MODULES_DIR

_LIB_CONFIG = Path(__file__).parent.parent / "lib" / "mv_track_bboxes.yaml"


def run_mv_track_bboxes(
    rgb_dir: str,
    weights_dir: str,
    output_dir: str,
    labeled_bbox_dir: str | None = None,
    config_path: str = str(_LIB_CONFIG),
    debug: int = -1,
    dev: bool = False,
) -> None:
    inputs = {
        "rgb_dir": rgb_dir,
        "weights_dir": weights_dir,
        "config_path": config_path,
    }
    if labeled_bbox_dir:
        inputs["labeled_bbox_dir"] = labeled_bbox_dir

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
    parser.add_argument("--rgb_dir", type=str, required=True, help="Directory containing input frames")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing model weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for outputs")
    parser.add_argument(
        "--labeled_bbox_dir", type=str, default=None,
        help="Optional directory with per-camera labeled object bbox JSONs "
             "(anchors primary person selection to the object bbox).",
    )
    parser.add_argument("--config_path", type=str, default=str(_LIB_CONFIG), help="Path to mv_track_bboxes.yaml")
    parser.add_argument("--debug", type=int, default=-1, help="Debug level")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_track_bboxes(
        rgb_dir=args.rgb_dir,
        weights_dir=args.weights_dir,
        output_dir=args.output_dir,
        labeled_bbox_dir=args.labeled_bbox_dir,
        config_path=args.config_path,
        debug=args.debug,
        dev=args.dev,
    )
