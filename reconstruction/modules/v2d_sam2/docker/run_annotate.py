import subprocess
import os

IMAGE_NAME = "v2d_sam2"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))

def run_annotate(video_path: str, prompts_path: str, port: int = 8080, dev: bool = False) -> None:
    video_path = os.path.abspath(video_path)
    prompts_path = os.path.abspath(prompts_path)

    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    prompts_dir = os.path.dirname(prompts_path)
    prompts_name = os.path.basename(prompts_path)

    os.makedirs(prompts_dir, exist_ok=True)

    cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-p", f"{port}:{port}",
        "-v", f"{video_dir}:/data/video",
        "-v", f"{prompts_dir}:/data/prompts",
    ]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.sam2.lib.annotate",
        "--video_path", f"/data/video/{video_name}",
        "--prompts_path", f"/data/prompts/{prompts_name}",
        "--port", str(port),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SAM2 annotation tool")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--prompts_path", type=str, required=True, help="Path to save prompts JSON")
    parser.add_argument("--port", type=int, default=8080, help="Port to run server on")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_annotate(args.video_path, args.prompts_path, args.port, dev=args.dev)
