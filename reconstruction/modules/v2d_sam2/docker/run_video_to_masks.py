# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.sam2.docker._config import IMAGE_NAME, MODULES_DIR


RUN_GENERATION_FILENAME = "run_generation.json"
OUTPUT_STATE_ABSENT = "absent"
OUTPUT_STATE_COMMITTED = "committed"
OUTPUT_STATE_INCOMPLETE = "incomplete"
_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def classify_existing_output(masks_dir: str) -> str:
    """Classify without trusting a commit marker's contents."""

    output = Path(os.path.abspath(masks_dir))
    if not os.path.lexists(output):
        return OUTPUT_STATE_ABSENT
    if output.is_dir() and (output / RUN_GENERATION_FILENAME).is_file():
        return OUTPUT_STATE_COMMITTED
    return OUTPUT_STATE_INCOMPLETE


def resolve_image_id(image: str = IMAGE_NAME) -> str:
    """Resolve a mutable local tag once, then execute its immutable image ID."""

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


def _rewrite_prompts_for_container(
    prompts_path: str,
) -> tuple[str, str]:
    """Stage prompts and referenced masks in one read-only mount tree."""

    with open(prompts_path) as f:
        data = json.load(f)
    prompts = data.get("prompts") or []
    tempdir = tempfile.mkdtemp(prefix="sam2_prompts_")
    try:
        source_root = os.path.dirname(os.path.abspath(prompts_path))
        staged_masks: dict[str, str] = {}
        for prompt in prompts:
            mask_path = prompt.get("mask_path")
            if not mask_path:
                continue
            source = (
                mask_path
                if os.path.isabs(mask_path)
                else os.path.join(source_root, mask_path)
            )
            source = os.path.abspath(source)
            relative = staged_masks.get(source)
            if relative is None:
                relative = os.path.join(
                    "prompt_assets",
                    f"{len(staged_masks):06d}",
                    os.path.basename(source),
                )
                destination = os.path.join(tempdir, relative)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copy2(source, destination)
                staged_masks[source] = relative
            prompt["mask_path"] = f"/data/prompts_path/{relative}"

        rewritten_path = os.path.join(tempdir, "prompts.json")
        with open(rewritten_path, "w") as f:
            json.dump(data, f, indent=2)
        return rewritten_path, tempdir
    except Exception:
        shutil.rmtree(tempdir, ignore_errors=True)
        raise


def run_video_to_masks(
    video_path: str,
    prompts_path: str,
    masks_dir: str,
    weights_dir: str,
    dev: bool = False,
    gpu: int = 0,
    *,
    image_id: str | None = None,
) -> None:
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError("gpu must be a non-negative physical GPU index")
    resolved_image_id = image_id or resolve_image_id()
    if not _IMAGE_ID_PATTERN.fullmatch(resolved_image_id):
        raise ValueError(
            "image_id must be an immutable sha256:<64 lowercase hex> Docker ID"
        )
    output = Path(os.path.abspath(masks_dir))
    output_state = classify_existing_output(str(output))
    if output_state == OUTPUT_STATE_INCOMPLETE:
        raise FileExistsError(
            f"Existing SAM2 output is not a committed generation; refusing "
            f"overwrite: {output}"
        )

    video_input = video_path
    input_directories = {"weights_dir"}
    input_files: set[str] = set()
    if os.path.isdir(video_path):
        # LazyFrameLoader also supports image directories. Mount the exact
        # directory root rather than regressing that established input mode.
        input_directories.add("video_path")
    else:
        if not os.path.isfile(video_path) and os.path.isfile(video_path + ".h5"):
            # Preserve LazyFrameLoader's extension-less HDF5 convenience while
            # still bind-mounting only the concrete source file.
            video_input = video_path + ".h5"
        input_files.add("video_path")
    rewritten_path, tempdir = _rewrite_prompts_for_container(
        prompts_path,
    )
    private_parent: Path | None = None
    try:
        inputs = {
            "video_path": video_input,
            "prompts_path": rewritten_path,
            "weights_dir": weights_dir,
        }
        outputs: dict[str, str] = {}
        atomic_outputs: set[str] = set()
        if output_state == OUTPUT_STATE_COMMITTED:
            inputs["masks_dir"] = str(output)
            input_directories.add("masks_dir")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            private_parent = Path(
                tempfile.mkdtemp(prefix=f".{output.name}.container.", dir=output.parent)
            )
            outputs["masks_dir"] = str(private_parent / output.name)
            atomic_outputs.add("masks_dir")

        run_in_container(
            image=resolved_image_id,
            module="v2d.sam2.lib.video_to_masks",
            inputs=inputs,
            outputs=outputs,
            extra_args={"image_id": resolved_image_id},
            dev=dev,
            modules_dir=MODULES_DIR,
            gpu_device=gpu,
            env={"CUDA_VISIBLE_DEVICES": "0"},
            network_disabled=True,
            strict_io_isolation=True,
            input_directories=input_directories,
            input_files=input_files,
            atomic_output_directories=atomic_outputs,
        )
        if private_parent is not None:
            staged_output = Path(outputs["masks_dir"])
            if (
                not staged_output.is_dir()
                or not (staged_output / RUN_GENERATION_FILENAME).is_file()
            ):
                raise RuntimeError(
                    "SAM2 container returned without a complete staged generation"
                )
            if os.path.lexists(output):
                raise FileExistsError(
                    f"SAM2 output appeared during generation; refusing overwrite: {output}"
                )
            os.replace(staged_output, output)
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)
        if private_parent is not None:
            shutil.rmtree(private_parent, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video to masks using SAM2")
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video"
    )
    parser.add_argument(
        "--prompts_path", type=str, required=True, help="Path to prompts JSON file"
    )
    parser.add_argument(
        "--masks_dir", type=str, required=True, help="Output directory for masks"
    )
    parser.add_argument(
        "--weights_dir", type=str, required=True, help="Path to SAM2 weights directory"
    )
    parser.add_argument(
        "--dev", action="store_true", help="Mount local modules for development"
    )
    parser.add_argument("--gpu", type=int, default=0, help="Physical host GPU index")
    parser.add_argument(
        "--image_id",
        help=(
            "Optional immutable Docker ID. Defaults to resolving the local "
            f"{IMAGE_NAME!r} tag once."
        ),
    )
    args = parser.parse_args()
    run_video_to_masks(
        args.video_path,
        args.prompts_path,
        args.masks_dir,
        args.weights_dir,
        dev=args.dev,
        gpu=args.gpu,
        image_id=args.image_id,
    )
