import subprocess
import os

IMAGE_NAME = "v2d_mesh"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_mesh_render_mask(
    mesh_path: str,
    intrinsics_path: str,
    output_mask_path: str,
    dev: bool = False,
) -> None:
    mesh_path = os.path.abspath(mesh_path)
    intrinsics_path = os.path.abspath(intrinsics_path)
    output_mask_path = os.path.abspath(output_mask_path)

    mesh_dir = os.path.dirname(mesh_path)
    mesh_name = os.path.basename(mesh_path)
    intrinsics_dir = os.path.dirname(intrinsics_path)
    intrinsics_name = os.path.basename(intrinsics_path)
    output_mask_dir = os.path.dirname(output_mask_path)
    output_mask_name = os.path.basename(output_mask_path)

    os.makedirs(output_mask_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{mesh_dir}:/data/mesh",
        "-v", f"{intrinsics_dir}:/data/intrinsics",
        "-v", f"{output_mask_dir}:/data/mask_out",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]

    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.mesh.lib.run_mesh_render_mask",
        "--mesh", f"/data/mesh/{mesh_name}",
        "--intrinsics", f"/data/intrinsics/{intrinsics_name}",
        "--output_mask", f"/data/mask_out/{output_mask_name}",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render a silhouette mask of a mesh (via Docker)")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--output_mask", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_mesh_render_mask(args.mesh, args.intrinsics, args.output_mask, dev=args.dev)
