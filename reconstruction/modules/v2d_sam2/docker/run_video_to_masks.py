# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import shutil
import tempfile

from v2d.docker.container import run_in_container
from v2d.sam2.docker._config import IMAGE_NAME, MODULES_DIR


def _rewrite_prompts_for_container(
    prompts_path: str,
) -> tuple[str, list[str], str | None]:
    """Rewrite host mask prompt paths to container-visible paths."""
    with open(prompts_path) as f:
        data = json.load(f)
    prompts = data.get("prompts") or []

    mask_dirs: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        mask_path = prompt.get("mask_path")
        if not mask_path:
            continue
        if not os.path.isabs(mask_path):
            mask_path = os.path.join(
                os.path.dirname(os.path.abspath(prompts_path)), mask_path
            )
        host_dir = os.path.dirname(os.path.abspath(mask_path))
        if host_dir not in seen:
            seen.add(host_dir)
            mask_dirs.append(host_dir)

    if not mask_dirs:
        return prompts_path, [], None

    host_to_container = {
        host_dir: f"/data/prompt_mask_dir_{i}"
        for i, host_dir in enumerate(mask_dirs)
    }
    extra_volumes = [
        f"{host_dir}:{container_dir}:ro"
        for host_dir, container_dir in host_to_container.items()
    ]

    for prompt in prompts:
        mask_path = prompt.get("mask_path")
        if not mask_path:
            continue
        if not os.path.isabs(mask_path):
            mask_path = os.path.join(
                os.path.dirname(os.path.abspath(prompts_path)), mask_path
            )
        host_dir = os.path.dirname(os.path.abspath(mask_path))
        prompt["mask_path"] = os.path.join(
            host_to_container[host_dir],
            os.path.basename(mask_path),
        )

    tempdir = tempfile.mkdtemp(prefix="sam2_prompts_")
    rewritten_path = os.path.join(
        tempdir,
        os.path.basename(prompts_path) or "prompts.json",
    )
    with open(rewritten_path, "w") as f:
        json.dump(data, f, indent=2)
    return rewritten_path, extra_volumes, tempdir


def run_video_to_masks(
    video_path: str,
    prompts_path: str,
    masks_dir: str,
    weights_dir: str,
    dev: bool = False,
) -> None:
    rewritten_path, extra_volumes, tempdir = _rewrite_prompts_for_container(
        prompts_path,
    )
    try:
        run_in_container(
            image=IMAGE_NAME,
            module="v2d.sam2.lib.video_to_masks",
            inputs={
                "video_path": video_path,
                "prompts_path": rewritten_path,
                "weights_dir": weights_dir,
            },
            outputs={"masks_dir": masks_dir},
            dev=dev,
            modules_dir=MODULES_DIR,
            gpus=True,
            extra_volumes=extra_volumes or None,
        )
    finally:
        if tempdir is not None:
            shutil.rmtree(tempdir, ignore_errors=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video to masks using SAM2")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--prompts_path", type=str, required=True, help="Path to prompts JSON file")
    parser.add_argument("--masks_dir", type=str, required=True, help="Output directory for masks")
    parser.add_argument("--weights_dir", type=str, required=True, help="Path to SAM2 weights directory")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_video_to_masks(args.video_path, args.prompts_path, args.masks_dir, args.weights_dir, dev=args.dev)
