from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.mv.postprocess.docker._config import IMAGE_NAME, MODULES_DIR

_LIB_CONFIG = Path(__file__).parent.parent / "lib" / "mv_estimate_ground_plane.yaml"


def run_mv_estimate_ground_plane(
    camera_params_path: str,
    depth_dir: str,
    human_pose_dir: str,
    output_dir: str,
    image_dir: str | None = None,
    mask_dir: str | None = None,
    config_path: str = str(_LIB_CONFIG),
    dev: bool = False,
) -> None:
    inputs = {
        "camera_params_path": camera_params_path,
        "depth_dir": depth_dir,
        "human_pose_dir": human_pose_dir,
        "config_path": config_path,
    }
    if image_dir is not None:
        inputs["image_dir"] = image_dir
    if mask_dir is not None:
        inputs["mask_dir"] = mask_dir
    outputs = {"output_dir": output_dir}

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mv.postprocess.lib.mv_estimate_ground_plane",
        inputs=inputs,
        outputs=outputs,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
        env={"PYTHONUNBUFFERED": "1"},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Estimate ground plane from multiview depth")
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--human_pose_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=str(_LIB_CONFIG))
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_estimate_ground_plane(
        camera_params_path=args.camera_params_path,
        depth_dir=args.depth_dir,
        human_pose_dir=args.human_pose_dir,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        config_path=args.config_path,
        dev=args.dev,
    )
