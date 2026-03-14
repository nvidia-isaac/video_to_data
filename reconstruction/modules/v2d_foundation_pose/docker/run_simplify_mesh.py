import subprocess
import os

IMAGE_NAME = "v2d_foundation_pose"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_simplify_mesh(
    input_mesh: str,
    output_mesh: str,
    faces: int = None,
    factor: float = None,
    dev: bool = False,
) -> None:
    input_mesh = os.path.abspath(input_mesh)
    output_mesh = os.path.abspath(output_mesh)

    input_dir, input_name = os.path.dirname(input_mesh), os.path.basename(input_mesh)
    output_dir, output_name = os.path.dirname(output_mesh), os.path.basename(output_mesh)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{input_dir}:/data/input",
        "-v", f"{output_dir}:/data/output",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.foundation_pose.lib.simplify_mesh",
        "--input-mesh", f"/data/input/{input_name}",
        "--output-mesh", f"/data/output/{output_name}",
    ]
    if faces is not None:
        cmd += ["--faces", str(faces)]
    if factor is not None:
        cmd += ["--factor", str(factor)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run mesh simplification in Docker")
    parser.add_argument("--input-mesh", required=True)
    parser.add_argument("--output-mesh", required=True)
    parser.add_argument("--faces", type=int, default=None)
    parser.add_argument("--factor", type=float, default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_simplify_mesh(
        args.input_mesh, args.output_mesh,
        faces=args.faces, factor=args.factor, dev=args.dev,
    )
