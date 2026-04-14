from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.mv.calibration.docker._config import IMAGE_NAME, MODULES_DIR

_DEFAULT_CONFIG = Path(__file__).parent.parent / "lib" / "calibrate_extrinsics.yaml"


def run_calibrate_extrinsics(
    camera_params_path: str,
    image_dir: str,
    output_dir: str,
    config_path: str = str(_DEFAULT_CONFIG),
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    num_workers: int | None = None,
    dev: bool = False,
) -> None:
    inputs = {
        "camera_params_path": camera_params_path,
        "image_dir": image_dir,
        "config_path": config_path,
    }

    outputs = {
        "output_dir": output_dir,
    }

    extra_args = {}
    if start is not None:
        extra_args["start"] = start
    if stop is not None:
        extra_args["stop"] = stop
    if step is not None:
        extra_args["step"] = step
    if num_workers is not None:
        extra_args["num_workers"] = num_workers

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mv.calibration.lib.calibrate_extrinsics",
        inputs=inputs,
        outputs=outputs,
        extra_args=extra_args,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run extrinsic calibration in Docker")
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=str(_DEFAULT_CONFIG))
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_calibrate_extrinsics(
        camera_params_path=args.camera_params_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        start=args.start,
        stop=args.stop,
        step=args.step,
        num_workers=args.num_workers,
        dev=args.dev,
    )
