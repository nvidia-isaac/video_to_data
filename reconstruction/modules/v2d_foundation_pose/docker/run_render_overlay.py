import os
from v2d.docker.container import run_in_container

IMAGE_NAME = "v2d_foundation_pose"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_render_overlay(
    video_path: str,
    poses_dir: str,
    mesh_path: str,
    camera_intrinsics_path: str,
    output_dir: str,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.render_overlay",
        inputs={"video_path": video_path, "poses_dir": poses_dir, "mesh_path": mesh_path, "camera_intrinsics_path": camera_intrinsics_path},
        outputs={"output_dir": output_dir},
        dev=dev,
        modules_dir=_MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FoundationPose render overlay in Docker")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--poses_dir", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--camera_intrinsics_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_render_overlay(
        args.video_path, args.poses_dir, args.mesh_path,
        args.camera_intrinsics_path, args.output_dir, dev=args.dev,
    )
