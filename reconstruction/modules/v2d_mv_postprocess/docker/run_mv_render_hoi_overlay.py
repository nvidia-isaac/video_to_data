from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.mv.postprocess.docker._config import IMAGE_NAME, MODULES_DIR

_LIB_CONFIG = Path(__file__).parent.parent / "lib" / "mv_render_hoi_overlay.yaml"


def _validate_overlay_inputs(
    object_mesh_path: str | None,
    object_pose_dir: str | None,
    human_pose_dir: str | None,
) -> None:
    if (object_mesh_path is None) != (object_pose_dir is None):
        raise ValueError(
            "Object overlay requires both object_mesh_path and object_pose_dir."
        )
    if object_mesh_path is None and human_pose_dir is None:
        raise ValueError(
            "HOI overlay requires at least one mesh source: provide object "
            "assets, human assets, or both."
        )


def run_mv_render_hoi_overlay(
    camera_params_path: str,
    rgb_dir: str,
    object_mesh_path: str | None = None,
    object_pose_dir: str | None = None,
    human_pose_dir: str | None = None,
    output_dir: str | None = None,
    config_path: str = str(_LIB_CONFIG),
    dev: bool = False,
) -> None:
    if output_dir is None:
        raise ValueError("output_dir is required.")
    _validate_overlay_inputs(
        object_mesh_path=object_mesh_path,
        object_pose_dir=object_pose_dir,
        human_pose_dir=human_pose_dir,
    )

    inputs = {
        "camera_params_path": camera_params_path,
        "rgb_dir": rgb_dir,
        "config_path": config_path,
    }
    if object_mesh_path is not None:
        inputs["object_mesh_path"] = object_mesh_path
    if object_pose_dir is not None:
        inputs["object_pose_dir"] = object_pose_dir
    if human_pose_dir is not None:
        inputs["human_pose_dir"] = human_pose_dir

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
    parser.add_argument("--rgb_dir", type=str, required=True, help="Directory containing input frames")
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--object_mesh_path", type=str)
    parser.add_argument("--object_pose_dir", type=str)
    parser.add_argument("--human_pose_dir", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=str(_LIB_CONFIG))
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_render_hoi_overlay(
        camera_params_path=args.camera_params_path,
        rgb_dir=args.rgb_dir,
        output_dir=args.output_dir,
        object_mesh_path=args.object_mesh_path,
        object_pose_dir=args.object_pose_dir,
        human_pose_dir=args.human_pose_dir,
        config_path=args.config_path,
        dev=args.dev,
    )
