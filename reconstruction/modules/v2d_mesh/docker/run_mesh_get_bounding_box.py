import subprocess
import os

IMAGE_NAME = "v2d_mesh"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_mesh_get_bounding_box(
    mesh_path: str,
    output_path: str,
    dev: bool = False,
) -> None:
    mesh_path = os.path.abspath(mesh_path)
    output_path = os.path.abspath(output_path)

    mesh_dir = os.path.dirname(mesh_path)
    mesh_name = os.path.basename(mesh_path)
    output_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{mesh_dir}:/data/mesh",
        "-v", f"{output_dir}:/data/output",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]

    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.mesh.lib.run_mesh_get_bounding_box",
        "--mesh", f"/data/mesh/{mesh_name}",
        "--output", f"/data/output/{output_name}",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute mesh bounding box (via Docker)")
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True, help="Output JSON file for BoundingBox3d")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_mesh_get_bounding_box(args.mesh, args.output, dev=args.dev)
