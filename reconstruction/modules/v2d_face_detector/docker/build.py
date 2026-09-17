import os
import subprocess

IMAGE_NAME = "v2d_face_detector"


def build_docker_image() -> None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    modules_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            IMAGE_NAME,
            "-f",
            os.path.join(current_dir, "Dockerfile"),
            modules_dir,
        ],
        check=True,
    )


if __name__ == "__main__":
    build_docker_image()
