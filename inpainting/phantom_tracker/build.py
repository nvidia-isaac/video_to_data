"""Build the pinned Blackwell-capable Phantom inference image."""

from __future__ import annotations

import subprocess
from pathlib import Path

from . import IMAGE_NAME


def build() -> None:
    stage = Path(__file__).resolve().parent
    subprocess.run(
        [
            "docker",
            "build",
            "--pull=false",
            "--tag",
            IMAGE_NAME,
            "--file",
            str(stage / "Dockerfile"),
            str(stage),
        ],
        check=True,
    )


if __name__ == "__main__":
    build()
