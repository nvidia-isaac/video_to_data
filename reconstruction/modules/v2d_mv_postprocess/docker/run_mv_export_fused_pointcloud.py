from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.mv.postprocess.docker._config import IMAGE_NAME, MODULES_DIR

_LIB_CONFIG = Path(__file__).parent.parent / "lib" / "mv_export_fused_pointcloud.yaml"


def run_mv_export_fused_pointcloud(
    camera_params_path: str,
    depth_dir: str,
    image_dir: str,
    output_dir: str,
    mask_dir: str | None = None,
    config_path: str = str(_LIB_CONFIG),
    dev: bool = False,
) -> None:
    inputs = {
        "camera_params_path": camera_params_path,
        "depth_dir": depth_dir,
        "image_dir": image_dir,
        "config_path": config_path,
    }
    if mask_dir is not None:
        inputs["mask_dir"] = mask_dir
    outputs = {"output_dir": output_dir}

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mv.postprocess.lib.mv_export_fused_pointcloud",
        inputs=inputs,
        outputs=outputs,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
        env={"PYTHONUNBUFFERED": "1"},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export fused multiview point clouds as PLY")
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--depth_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=str(_LIB_CONFIG))
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_export_fused_pointcloud(
        camera_params_path=args.camera_params_path,
        depth_dir=args.depth_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        mask_dir=args.mask_dir,
        config_path=args.config_path,
        dev=args.dev,
    )
