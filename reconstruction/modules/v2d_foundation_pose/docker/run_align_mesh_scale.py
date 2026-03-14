import subprocess
import os

IMAGE_NAME = "v2d_foundation_pose"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_align_mesh_scale(
    mesh_path: str,
    depth_path: str,
    mask_path: str,
    intrinsics_path: str,
    transform_path: str,
    output_transform_path: str,
    dev: bool = False,
) -> None:
    mesh_path = os.path.abspath(mesh_path)
    depth_path = os.path.abspath(depth_path)
    mask_path = os.path.abspath(mask_path)
    intrinsics_path = os.path.abspath(intrinsics_path)
    transform_path = os.path.abspath(transform_path)
    output_transform_path = os.path.abspath(output_transform_path)

    mesh_dir, mesh_name = os.path.dirname(mesh_path), os.path.basename(mesh_path)
    depth_dir, depth_name = os.path.dirname(depth_path), os.path.basename(depth_path)
    mask_dir, mask_name = os.path.dirname(mask_path), os.path.basename(mask_path)
    intrinsics_dir, intrinsics_name = os.path.dirname(intrinsics_path), os.path.basename(intrinsics_path)
    transform_dir, transform_name = os.path.dirname(transform_path), os.path.basename(transform_path)
    output_dir, output_name = os.path.dirname(output_transform_path), os.path.basename(output_transform_path)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{mesh_dir}:/data/mesh",
        "-v", f"{depth_dir}:/data/depth",
        "-v", f"{mask_dir}:/data/mask",
        "-v", f"{intrinsics_dir}:/data/intrinsics",
        "-v", f"{transform_dir}:/data/transform",
        "-v", f"{output_dir}:/data/output",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.foundation_pose.lib.align_mesh_scale",
        "--mesh", f"/data/mesh/{mesh_name}",
        "--depth", f"/data/depth/{depth_name}",
        "--mask", f"/data/mask/{mask_name}",
        "--intrinsics", f"/data/intrinsics/{intrinsics_name}",
        "--transform", f"/data/transform/{transform_name}",
        "--output-transform", f"/data/output/{output_name}",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run mesh scale alignment in Docker")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--output-transform", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_align_mesh_scale(
        args.mesh, args.depth, args.mask, args.intrinsics,
        args.transform, args.output_transform, dev=args.dev,
    )
