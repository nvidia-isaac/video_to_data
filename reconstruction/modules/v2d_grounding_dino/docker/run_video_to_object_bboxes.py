import subprocess
import os

IMAGE_NAME = "v2d_grounding_dino"

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODULES_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))


def run_video_to_object_bboxes(
    video_path: str,
    output_path: str,
    prompt: str,
    model_dir: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    debug_output: str = None,
    dev: bool = False,
) -> None:
    video_path = os.path.abspath(video_path)
    output_path = os.path.abspath(output_path)
    model_dir = os.path.abspath(model_dir)

    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    output_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "docker", "run", "-it", "--rm",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-v", f"{video_dir}:/data/video",
        "-v", f"{output_dir}:/data/output",
        "-v", f"{model_dir}:/data/models",
    ]
    if debug_output:
        debug_output = os.path.abspath(debug_output)
        os.makedirs(debug_output, exist_ok=True)
        cmd += ["-v", f"{debug_output}:/data/debug"]
    if dev:
        cmd += ["-v", f"{_MODULES_DIR}:/workspace"]
    cmd += [
        IMAGE_NAME,
        "python", "-m", "v2d.grounding_dino.lib.video_to_object_bboxes",
        "--video_path", f"/data/video/{video_name}",
        "--output_path", f"/data/output/{output_name}",
        "--prompt", prompt,
        "--model_dir", "/data/models",
        "--box_threshold", str(box_threshold),
        "--text_threshold", str(text_threshold),
    ]
    if debug_output:
        cmd += ["--debug_output", "/data/debug"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grounding DINO: detect objects in video frames")
    parser.add_argument('--video_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--box_threshold', type=float, default=0.35)
    parser.add_argument('--text_threshold', type=float, default=0.25)
    parser.add_argument('--debug_output', type=str, default=None,
                        help='Directory to save annotated debug frames')
    parser.add_argument('--dev', action='store_true', help='Mount local modules for development')
    args = parser.parse_args()
    run_video_to_object_bboxes(
        args.video_path, args.output_path, args.prompt, args.model_dir,
        box_threshold=args.box_threshold, text_threshold=args.text_threshold,
        debug_output=args.debug_output, dev=args.dev,
    )
