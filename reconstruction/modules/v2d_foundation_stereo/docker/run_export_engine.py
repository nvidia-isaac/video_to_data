import subprocess
import os

IMAGE_NAME = "v2d_foundation_stereo"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_export_engine(model_dir: str, dev: bool = False) -> None:
    model_dir = os.path.abspath(model_dir)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{model_dir}:/data/models",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-c",
        "from v2d.foundation_stereo.lib.export_engine import ensure_engine; ensure_engine('/data/models')",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Foundation Stereo ONNX to TensorRT engine")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the ONNX model (engine will be saved here)")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_export_engine(model_dir=args.model_dir, dev=args.dev)
