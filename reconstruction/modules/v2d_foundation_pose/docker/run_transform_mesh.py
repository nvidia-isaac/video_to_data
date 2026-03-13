import subprocess
import os

IMAGE_NAME = "v2d_foundation_pose"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_transform_mesh(
    input_mesh: str,
    output_mesh: str,
    transform_path: str,
    dev: bool = False,
) -> None:
    input_mesh = os.path.abspath(input_mesh)
    output_mesh = os.path.abspath(output_mesh)
    transform_path = os.path.abspath(transform_path)

    input_dir, input_name = os.path.dirname(input_mesh), os.path.basename(input_mesh)
    output_dir, output_name = os.path.dirname(output_mesh), os.path.basename(output_mesh)
    transform_dir, transform_name = os.path.dirname(transform_path), os.path.basename(transform_path)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "docker", "run", "-it", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{input_dir}:/data/input",
        "-v", f"{output_dir}:/data/output",
        "-v", f"{transform_dir}:/data/transform",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.foundation_pose.lib.transform_mesh",
        "--input-mesh", f"/data/input/{input_name}",
        "--output-mesh", f"/data/output/{output_name}",
        "--transform", f"/data/transform/{transform_name}",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run mesh transform in Docker")
    parser.add_argument("--input-mesh", required=True)
    parser.add_argument("--output-mesh", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_transform_mesh(
        args.input_mesh, args.output_mesh, args.transform, dev=args.dev,
    )
