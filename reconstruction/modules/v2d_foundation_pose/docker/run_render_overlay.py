import subprocess
import os

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
    video_path = os.path.abspath(video_path)
    poses_dir = os.path.abspath(poses_dir)
    mesh_path = os.path.abspath(mesh_path)
    camera_intrinsics_path = os.path.abspath(camera_intrinsics_path)
    output_dir = os.path.abspath(output_dir)

    video_dir, video_name = os.path.dirname(video_path), os.path.basename(video_path)
    intrinsics_dir, intrinsics_name = os.path.dirname(camera_intrinsics_path), os.path.basename(camera_intrinsics_path)
    mesh_dir, mesh_name = os.path.dirname(mesh_path), os.path.basename(mesh_path)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{video_dir}:/data/video",
        "-v", f"{poses_dir}:/data/poses",
        "-v", f"{mesh_dir}:/data/mesh",
        "-v", f"{intrinsics_dir}:/data/intrinsics",
        "-v", f"{output_dir}:/data/output",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.foundation_pose.lib.render_overlay",
        "--video_path", f"/data/video/{video_name}",
        "--poses_dir", "/data/poses",
        "--mesh_path", f"/data/mesh/{mesh_name}",
        "--camera_intrinsics_path", f"/data/intrinsics/{intrinsics_name}",
        "--output_dir", "/data/output",
    ]
    subprocess.run(cmd, check=True)


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
