from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.foundation_stereo.docker._config import IMAGE_NAME, MODULES_DIR

_LIB_CONFIG = Path(__file__).parent.parent / "lib" / "mv_image_list_to_depth.yaml"


def run_mv_image_list_to_depth(
    camera_params_path: str,
    rgb_dir: str,
    output_dir: str,
    model_dir: str,
    scale: float | None = None,
    config_path: str = str(_LIB_CONFIG),
    dev: bool = False,
) -> None:
    inputs = {
        "camera_params_path": camera_params_path,
        "rgb_dir": rgb_dir,
        "config_path": config_path,
    }
    outputs = {
        "output_dir": output_dir,
        "model_dir": model_dir,
    }

    extra_args: dict = {}
    if scale is not None:
        extra_args["scale"] = scale

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_stereo.lib.mv_image_list_to_depth",
        inputs=inputs,
        outputs=outputs,
        extra_args=extra_args or None,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={"PYTHONUNBUFFERED": "1"},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-view stereo depth estimation with Foundation Stereo TRT"
    )
    parser.add_argument("--camera_params_path", type=str, required=True,
                        help="Path to EDEX file with camera calibration")
    parser.add_argument("--rgb_dir", type=str, required=True,
                        help="Root directory containing per-camera image folders")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for depth and intrinsics")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing Foundation Stereo ONNX/engine")
    parser.add_argument("--scale", type=float, default=None,
                        help="Scale factor for output resolution (e.g. 0.5 for half)")
    parser.add_argument("--config_path", type=str, default=str(_LIB_CONFIG),
                        help="Path to config YAML")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_image_list_to_depth(
        camera_params_path=args.camera_params_path,
        rgb_dir=args.rgb_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        scale=args.scale,
        config_path=args.config_path,
        dev=args.dev,
    )
