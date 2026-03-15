import os
from v2d.docker.container import run_in_container
from v2d.foundation_pose.docker._config import IMAGE_NAME, MODULES_DIR


def run_video_to_poses(
    video_path: str,
    depth_folder: str,
    masks_folder: str,
    camera_intrinsics_path: str,
    mesh_path: str,
    poses_dir: str,
    weights_dir: str,
    reference_frame: int = 0,
    target_width: int = None,
    target_height: int = None,
    debug_dir: str = None,
    dev: bool = False,
) -> None:
    weights_abs = os.path.abspath(weights_dir)
    weights_container = f"/data/weights_dir/{os.path.basename(weights_abs)}"
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.foundation_pose.lib.video_to_poses",
        inputs={"video_path": video_path, "depth_folder": depth_folder, "masks_folder": masks_folder, "camera_intrinsics_path": camera_intrinsics_path, "mesh_path": mesh_path, "weights_dir": weights_dir},
        outputs={"poses_dir": poses_dir, "debug_dir": debug_dir},
        extra_args={"reference_frame": reference_frame, "target_width": target_width, "target_height": target_height},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={"FOUNDATIONPOSE_WEIGHTS_DIR": weights_container},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FoundationPose video to poses in Docker")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--depth_folder", required=True)
    parser.add_argument("--masks_folder", required=True)
    parser.add_argument("--camera_intrinsics_path", required=True)
    parser.add_argument("--mesh_path", required=True)
    parser.add_argument("--poses_dir", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--reference_frame", type=int, default=0)
    parser.add_argument("--target_width", type=int, default=None)
    parser.add_argument("--target_height", type=int, default=None)
    parser.add_argument("--debug_dir", type=str, default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_video_to_poses(
        args.video_path, args.depth_folder, args.masks_folder,
        args.camera_intrinsics_path, args.mesh_path, args.poses_dir,
        args.weights_dir, reference_frame=args.reference_frame,
        target_width=args.target_width, target_height=args.target_height,
        debug_dir=args.debug_dir, dev=args.dev,
    )
