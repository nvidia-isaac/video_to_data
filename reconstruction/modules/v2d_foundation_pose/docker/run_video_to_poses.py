import subprocess
import os

IMAGE_NAME = "v2d_foundation_pose"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


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
    video_path = os.path.abspath(video_path)
    depth_folder = os.path.abspath(depth_folder)
    masks_folder = os.path.abspath(masks_folder)
    camera_intrinsics_path = os.path.abspath(camera_intrinsics_path)
    mesh_path = os.path.abspath(mesh_path)
    poses_dir = os.path.abspath(poses_dir)
    weights_dir = os.path.abspath(weights_dir)

    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    intrinsics_dir = os.path.dirname(camera_intrinsics_path)
    intrinsics_name = os.path.basename(camera_intrinsics_path)
    mesh_dir = os.path.dirname(mesh_path)
    mesh_name = os.path.basename(mesh_path)

    os.makedirs(poses_dir, exist_ok=True)

    cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-e", "FOUNDATIONPOSE_WEIGHTS_DIR=/data/weights",
        "-v", f"{video_dir}:/data/video",
        "-v", f"{depth_folder}:/data/depth",
        "-v", f"{masks_folder}:/data/masks",
        "-v", f"{intrinsics_dir}:/data/intrinsics",
        "-v", f"{mesh_dir}:/data/mesh",
        "-v", f"{poses_dir}:/data/poses",
        "-v", f"{weights_dir}:/data/weights",
    ]
    if debug_dir:
        debug_dir = os.path.abspath(debug_dir)
        os.makedirs(debug_dir, exist_ok=True)
        cmd += ["-v", f"{debug_dir}:/data/debug"]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.foundation_pose.lib.video_to_poses",
        "--video_path", f"/data/video/{video_name}",
        "--depth_folder", "/data/depth",
        "--masks_folder", "/data/masks",
        "--camera_intrinsics_path", f"/data/intrinsics/{intrinsics_name}",
        "--mesh_path", f"/data/mesh/{mesh_name}",
        "--poses_dir", "/data/poses",
        "--reference_frame", str(reference_frame),
    ]
    if target_width is not None:
        cmd += ["--target_width", str(target_width)]
    if target_height is not None:
        cmd += ["--target_height", str(target_height)]
    if debug_dir:
        cmd += ["--debug_dir", "/data/debug"]
    subprocess.run(cmd, check=True)


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
