from v2d.docker.container import run_in_container
from v2d.sam3d_body.docker._config import IMAGE_NAME, MODULES_DIR


def run_estimate_mhr_params(
    cam_intrinsics_path: str,
    weights_dir: str,
    bbox_path: str,
    output_params_path: str,
    image_dir: str | None = None,
    video_path: str | None = None,
    output_mesh_path: str | None = None,
    debug: int = -1,
    dev: bool = False,
) -> None:
    inputs = {
        "cam_intrinsics_path": cam_intrinsics_path,
        "weights_dir": weights_dir,
        "bbox_path": bbox_path,
    }
    if image_dir:
        inputs["image_dir"] = image_dir
    if video_path:
        inputs["video_path"] = video_path

    outputs = {"output_params_path": output_params_path}
    if output_mesh_path:
        outputs["output_mesh_path"] = output_mesh_path

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
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SAM3D-Body MHR estimation (single camera)")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_dir", type=str)
    input_group.add_argument("--video_path", type=str)

    parser.add_argument("--cam_intrinsics_path", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--bbox_path", type=str, required=True)
    parser.add_argument("--output_params_path", type=str, required=True)
    parser.add_argument("--output_mesh_path", type=str, default=None)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_estimate_mhr_params(
        cam_intrinsics_path=args.cam_intrinsics_path,
        weights_dir=args.weights_dir,
        bbox_path=args.bbox_path,
        output_params_path=args.output_params_path,
        image_dir=args.image_dir,
        video_path=args.video_path,
        output_mesh_path=args.output_mesh_path,
        debug=args.debug,
        dev=args.dev,
    )
