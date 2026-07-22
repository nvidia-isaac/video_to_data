# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.wilor.docker._config import IMAGE_NAME, MODULES_DIR


_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
RUN_GENERATION_FILENAME = "run_generation.json"
OUTPUT_STATE_COMMITTED = "committed"
# Retained as a public compatibility symbol for callers that imported the old
# classifier state.  New classification deliberately never returns it: a
# pre-manifest directory cannot be distinguished safely from an interrupted
# incremental generation.
OUTPUT_STATE_LEGACY = "legacy"
OUTPUT_STATE_INCOMPLETE = "absent_or_incomplete"


def classify_existing_output(output_dir: str, reference_frame: int) -> str:
    """Classify host output without treating a manifest as validated.

    A committed-looking output must still be passed to ``video_to_hands`` so
    its strict generation validator can verify source identity, expected files,
    and output hashes.  Pre-manifest directories always remain incomplete:
    the presence of one reference-frame JSON cannot prove that an older
    incremental run reached its final frame.
    """

    # Keep the established call signature even though safe classification no
    # longer depends on one arbitrarily selected frame.
    del reference_frame
    output = Path(output_dir)
    manifest = output / RUN_GENERATION_FILENAME
    if manifest.is_file():
        return OUTPUT_STATE_COMMITTED
    return OUTPUT_STATE_INCOMPLETE


def resolve_image_id(image: str = IMAGE_NAME) -> str:
    """Resolve a local tag once, then execute only its immutable image ID."""

    completed = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    image_id = completed.stdout.strip()
    if not _IMAGE_ID_PATTERN.fullmatch(image_id):
        raise RuntimeError(
            f"docker image inspect returned an invalid immutable ID for {image!r}: "
            f"{image_id!r}"
        )
    return image_id


def run_video_to_hands(
    video_path: str,
    output_dir: str,
    weights_dir: str,
    bboxes_dir: str | None = None,
    dev: bool = False,
    *,
    image_id: str | None = None,
    gpu: int = 0,
) -> None:
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError("gpu must be a non-negative physical GPU index")
    resolved_image_id = image_id or resolve_image_id()
    if not _IMAGE_ID_PATTERN.fullmatch(resolved_image_id):
        raise ValueError(
            "image_id must be an immutable sha256:<64 lowercase hex> Docker ID"
        )

    output = Path(os.path.abspath(output_dir))
    output_exists = os.path.lexists(output)
    if output_exists and not output.is_dir():
        raise FileExistsError(f"WiLoR output path is not a directory: {output}")

    inputs = {
        "video_path": video_path,
        "weights_dir": weights_dir,
    }
    if bboxes_dir is not None:
        inputs["bboxes_dir"] = bboxes_dir
    input_directories = {"weights_dir"}
    if bboxes_dir is not None:
        input_directories.add("bboxes_dir")

    # Existing generations need validation only, so expose them read-only as
    # an input even though the in-container CLI calls the path ``output_dir``.
    # New generations are written beneath a private host directory. The host
    # publishes the validated directory afterward, keeping the container from
    # receiving a writable mount of the pipeline run root and all its siblings.
    private_parent: Path | None = None
    container_outputs: dict[str, str] = {}
    atomic_outputs: set[str] = set()
    if output_exists:
        inputs["output_dir"] = str(output)
        input_directories.add("output_dir")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        private_parent = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.container.",
                dir=output.parent,
            )
        )
        container_outputs["output_dir"] = str(private_parent / output.name)
        atomic_outputs.add("output_dir")

    try:
        run_in_container(
            image=resolved_image_id,
            module="v2d.wilor.lib.video_to_hands",
            inputs=inputs,
            outputs=container_outputs,
            extra_args={"image_id": resolved_image_id},
            dev=dev,
            modules_dir=MODULES_DIR,
            gpu_device=gpu,
            env={"CUDA_VISIBLE_DEVICES": "0"},
            network_disabled=True,
            strict_io_isolation=True,
            input_directories=input_directories,
            input_files={"video_path"},
            atomic_output_directories=atomic_outputs,
        )
        if private_parent is not None:
            staged_output = Path(container_outputs["output_dir"])
            if not staged_output.is_dir() or not (
                staged_output / RUN_GENERATION_FILENAME
            ).is_file():
                raise RuntimeError(
                    "WiLoR container returned without a complete staged generation"
                )
            if os.path.lexists(output):
                raise FileExistsError(
                    f"WiLoR output appeared during generation; refusing overwrite: {output}"
                )
            os.replace(staged_output, output)
    finally:
        if private_parent is not None:
            shutil.rmtree(private_parent, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WiLoR over a video")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument(
        "--image_id",
        help=(
            "Optional immutable Docker ID. When omitted, resolve the local "
            f"{IMAGE_NAME!r} tag once and execute the resulting ID."
        ),
    )
    parser.add_argument("--bboxes_dir", default=None)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="Physical host GPU index")
    args = parser.parse_args()
    run_video_to_hands(
        video_path=args.video_path,
        output_dir=args.output_dir,
        weights_dir=args.weights_dir,
        image_id=args.image_id,
        bboxes_dir=args.bboxes_dir,
        dev=args.dev,
        gpu=args.gpu,
    )
