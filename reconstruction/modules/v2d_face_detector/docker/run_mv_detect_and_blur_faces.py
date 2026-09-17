"""Run multiview YuNet face detection and blurring in Docker."""

from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.face_detector.docker._config import IMAGE_NAME, MODULES_DIR


_DEFAULT_CONFIG = (
    Path(__file__).parent.parent / "lib" / "mv_detect_and_blur_faces.yaml"
)


def run_mv_detect_and_blur_faces(
    rgb_dir: str,
    model_dir: str,
    output_dir: str,
    config_path: str = str(_DEFAULT_CONFIG),
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.face_detector.lib.mv_detect_and_blur_faces",
        inputs={
            "rgb_dir": rgb_dir,
            "model_dir": model_dir,
            "config_path": config_path,
        },
        outputs={"output_dir": output_dir},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=False,
        env={"PYTHONUNBUFFERED": "1"},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config_path", default=str(_DEFAULT_CONFIG))
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_mv_detect_and_blur_faces(
        rgb_dir=args.rgb_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        dev=args.dev,
    )
