import subprocess
from pathlib import Path

IMAGE_NAME = "v2d_cusfm"

_DOCKER_DIR = Path(__file__).parent
_MODULES_DIR = _DOCKER_DIR.parent.parent  # reconstruction/modules/


def build(tag: str = IMAGE_NAME) -> None:
    subprocess.run([
        "docker", "build",
        "-f", str(_DOCKER_DIR / "Dockerfile"),
        "-t", tag,
        str(_MODULES_DIR),
    ], check=True)


if __name__ == "__main__":
    build()
