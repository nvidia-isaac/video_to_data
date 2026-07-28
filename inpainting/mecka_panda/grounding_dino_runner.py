"""Run Grounding-DINO with an already resolved immutable Docker image ID."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def _runtime() -> tuple[Callable[..., None], str]:
    """Import reconstruction's container helper only inside its own environment."""
    from v2d.docker.container import run_in_container
    from v2d.grounding_dino.docker._config import MODULES_DIR

    return run_in_container, MODULES_DIR


def run_image_to_object_bboxes(
    *,
    image_path: str,
    output_path: str,
    prompt: str,
    model_dir: str,
    image_id: str,
    cache_dir: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    debug_output: str | None = None,
    dev: bool = False,
) -> None:
    """Execute the detector by immutable ID, never by its mutable local tag."""
    if _IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise ValueError(
            "image_id must be an immutable sha256:<64 lowercase hex> Docker ID"
        )
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    run_in_container, modules_dir = _runtime()
    run_in_container(
        image=image_id,
        module="v2d.grounding_dino.lib.image_to_object_bboxes",
        inputs={"image_path": image_path, "model_dir": model_dir},
        outputs={"output_path": output_path, "debug_output": debug_output},
        extra_args={
            "prompt": prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
        },
        dev=dev,
        modules_dir=modules_dir,
        gpus=True,
        # The container runs as the caller's numeric UID, which has no passwd
        # entry. Transformers otherwise resolves its cache beneath the
        # unwritable filesystem root (/.cache).
        env={
            "HOME": "/tmp",
            "HF_HOME": "/huggingface-cache",
            "TRANSFORMERS_CACHE": "/huggingface-cache/transformers",
        },
        extra_volumes=[f"{cache_path}:/huggingface-cache:rw"],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--image_id", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--debug_output")
    parser.add_argument("--dev", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    values: dict[str, Any] = vars(arguments)
    run_image_to_object_bboxes(**values)


if __name__ == "__main__":
    main()
