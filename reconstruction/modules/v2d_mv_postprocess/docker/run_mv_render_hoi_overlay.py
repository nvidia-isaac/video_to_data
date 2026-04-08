from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.mv.postprocess.docker._config import IMAGE_NAME, MODULES_DIR

_LIB_CONFIG = Path(__file__).parent.parent / "lib" / "mv_render_hoi_overlay.yaml"


def run_mv_render_hoi_overlay(
    camera_params_path: str,
    object_mesh_path: str,
    object_pose_dir: str,
    human_pose_dir: str,
    output_dir: str,
    image_dir: str | None = None,
    video_dir: str | None = None,
    config_path: str = str(_LIB_CONFIG),
    dev: bool = False,
) -> None:
    inputs = {
        "camera_params_path": camera_params_path,
        "object_mesh_path": object_mesh_path,
        "object_pose_dir": object_pose_dir,
        "human_pose_dir": human_pose_dir,
        "config_path": config_path,
    }
    if image_dir:
        inputs["image_dir"] = image_dir
    if video_dir:
        inputs["video_dir"] = video_dir

    outputs = {"output_dir": output_dir}

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.mv.postprocess.lib.mv_render_hoi_overlay",
        inputs=inputs,
        outputs=outputs,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={"PYTHONUNBUFFERED": "1"},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render HOI overlay videos")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_dir", type=str, help="Directory containing images")
    input_group.add_argument("--video_dir", type=str, help="Directory containing videos")

    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--object_mesh_path", type=str, required=True)
    parser.add_argument("--object_pose_dir", type=str, required=True)
    parser.add_argument("--human_pose_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=str(_LIB_CONFIG))
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_render_hoi_overlay(
        camera_params_path=args.camera_params_path,
        object_mesh_path=args.object_mesh_path,
        object_pose_dir=args.object_pose_dir,
        human_pose_dir=args.human_pose_dir,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        video_dir=args.video_dir,
        config_path=args.config_path,
        dev=args.dev,
    )
