from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.sam3d_body.docker._config import IMAGE_NAME, MODULES_DIR


def run_estimate_mhr_params(
    rgb_path: str,
    cam_intrinsics_path: str,
    weights_dir: str,
    bbox_path: str,
    output_params_path: str,
    output_mesh_path: str | None = None,
    debug: int = -1,
    dev: bool = False,
) -> None:
    inputs = {
        "rgb_path": rgb_path,
        "cam_intrinsics_path": cam_intrinsics_path,
        "weights_dir": weights_dir,
        "bbox_path": bbox_path,
    }

    outputs = {"output_params_path": output_params_path}
    if output_mesh_path:
        outputs["output_mesh_path"] = output_mesh_path

    weights_abs = Path(weights_dir).resolve()
    weights_container = f"/data/weights_dir/{weights_abs.name}"
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.sam3d_body.lib.estimate_mhr_params",
        inputs=inputs,
        outputs=outputs,
        extra_args={"debug": debug if debug >= 0 else None},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={
            "PYTHONUNBUFFERED": "1",
            "TORCH_HOME": f"{weights_container}/torch_home",
            "HF_HOME": f"{weights_container}/hf_home",
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SAM3D-Body MHR estimation (single camera)")
    parser.add_argument("--rgb_path", type=str, required=True,
                        help="Path to input frames (image dir, .h5, or video file)")
    parser.add_argument("--cam_intrinsics_path", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--bbox_path", type=str, required=True)
    parser.add_argument("--output_params_path", type=str, required=True)
    parser.add_argument("--output_mesh_path", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_estimate_mhr_params(
        rgb_path=args.rgb_path,
        cam_intrinsics_path=args.cam_intrinsics_path,
        weights_dir=args.weights_dir,
        bbox_path=args.bbox_path,
        output_params_path=args.output_params_path,
        output_mesh_path=args.output_mesh_path,
        debug=args.debug,
        dev=args.dev,
    )
