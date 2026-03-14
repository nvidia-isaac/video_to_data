import subprocess
import os

IMAGE_NAME = "v2d_mesh"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_mesh_transform(
    input_mesh_path: str,
    transform_path: str,
    output_mesh_path: str,
    dev: bool = False,
) -> None:
    input_mesh_path = os.path.abspath(input_mesh_path)
    transform_path = os.path.abspath(transform_path)
    output_mesh_path = os.path.abspath(output_mesh_path)

    input_mesh_dir = os.path.dirname(input_mesh_path)
    input_mesh_name = os.path.basename(input_mesh_path)
    transform_dir = os.path.dirname(transform_path)
    transform_name = os.path.basename(transform_path)
    output_mesh_dir = os.path.dirname(output_mesh_path)
    output_mesh_name = os.path.basename(output_mesh_path)

    os.makedirs(output_mesh_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{input_mesh_dir}:/data/input_mesh",
        "-v", f"{transform_dir}:/data/transform",
        "-v", f"{output_mesh_dir}:/data/output_mesh",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]

    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.mesh.lib.run_mesh_transform",
        "--input_mesh", f"/data/input_mesh/{input_mesh_name}",
        "--transform", f"/data/transform/{transform_name}",
        "--output_mesh", f"/data/output_mesh/{output_mesh_name}",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply a Transform3d to a mesh (via Docker)")
    parser.add_argument("--input_mesh", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--output_mesh", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_mesh_transform(args.input_mesh, args.transform, args.output_mesh, dev=args.dev)
