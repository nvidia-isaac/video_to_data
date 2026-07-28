# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run E2FGVI in its image using the repository's standard mount helper."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

from v2d.docker.container import run_in_container
from v2d.inpainting.e2fgvi.validation import (
    InferenceConfig,
    build_metadata,
    canonical_json,
    enrich_completed_metadata as enrich_legacy_completed_metadata,
    validate_output_paths,
    validate_run,
)

from ._config import IMAGE_NAME, LIB_SRC_DIR

CONTAINER_MODULE = "v2d.inpainting.e2fgvi.cli"


def default_metadata_path(output_video: str) -> str:
    return f"{output_video}.json"


def resolve_local_image_id(image: str) -> str:
    """Resolve one local image reference once to an immutable Docker ID."""

    if not image.strip():
        raise ValueError("container image must not be empty")
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"cannot resolve local Docker image {image!r}: {detail}")
    image_id = completed.stdout.strip()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise RuntimeError(
            f"docker returned invalid immutable image ID for {image!r}: {image_id!r}"
        )
    return image_id


def run_inpaint(
    input_video: str,
    masks: str,
    checkpoint: str,
    output_video: str,
    metadata_path: str | None = None,
    *,
    downscale: float = 1.0,
    max_size: int = 0,
    dilation_iterations: int = 4,
    dilation_kernel: int = 3,
    neighbor_stride: int = 5,
    ref_stride: int = 20,
    num_ref: int = -1,
    device: str = "cuda:0",
    gpu: int = 0,
    codec: str = "mp4v",
    seed: int = 0,
    validate_only: bool = False,
    enrich_completed_metadata: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
    dev: bool = False,
) -> dict[str, Any] | None:
    """Validate locally or run the container on one explicitly selected GPU."""
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError("gpu must be a non-negative physical GPU index")
    if device.startswith("cuda") and device not in {"cuda", "cuda:0"}:
        raise ValueError(
            "the host runner exposes the selected physical GPU as cuda:0; "
            "use device='cuda:0' and choose it with gpu=<physical index>"
        )
    metadata_path = metadata_path or default_metadata_path(output_video)
    validate_output_paths(input_video, masks, checkpoint, output_video, metadata_path)
    if validate_only and enrich_completed_metadata:
        raise ValueError(
            "validate_only and enrich_completed_metadata are mutually exclusive"
        )
    if dry_run and enrich_completed_metadata:
        raise ValueError("dry_run and enrich_completed_metadata are mutually exclusive")
    if enrich_completed_metadata:
        report = enrich_legacy_completed_metadata(
            metadata_path,
            input_video,
            masks,
            checkpoint,
            output_video,
        )
        print(canonical_json(report), end="")
        return report
    config = InferenceConfig(
        downscale=downscale,
        max_size=max_size,
        dilation_iterations=dilation_iterations,
        dilation_kernel=dilation_kernel,
        neighbor_stride=neighbor_stride,
        ref_stride=ref_stride,
        num_ref=num_ref,
        device=device,
        codec=codec,
        seed=seed,
    )

    if dry_run:
        if not overwrite:
            intended_outputs = [metadata_path] if validate_only else [output_video, metadata_path]
            existing = [str(Path(path).resolve()) for path in intended_outputs if Path(path).exists()]
            if existing:
                raise FileExistsError(
                    f"refusing to overwrite existing output(s): {', '.join(existing)}"
                )
        plan = validate_run(input_video, masks, checkpoint, config)
        report = build_metadata(
            plan,
            input_video,
            masks,
            checkpoint,
            output_video,
            "validated",
        )
        print(canonical_json(report), end="")
        return report

    use_gpu = device.startswith("cuda") and not validate_only
    # Docker exposes only the selected physical GPU. CUDA reindexes that device
    # inside the container, so inference always addresses it as cuda:0.
    environment = {"CUDA_VISIBLE_DEVICES": "0"} if use_gpu else None
    image_id = resolve_local_image_id(IMAGE_NAME)
    run_in_container(
        image=image_id,
        module=CONTAINER_MODULE,
        inputs={
            "input_video": input_video,
            "masks": masks,
            "checkpoint": checkpoint,
        },
        outputs={
            "output_video": output_video,
            "metadata_path": metadata_path,
        },
        extra_args={
            "downscale": downscale,
            "max_size": max_size,
            "dilation_iterations": dilation_iterations,
            "dilation_kernel": dilation_kernel,
            "neighbor_stride": neighbor_stride,
            "ref_stride": ref_stride,
            "num_ref": num_ref,
            "device": device,
            "codec": codec,
            "seed": seed,
            "validate_only": validate_only,
            "container_image": IMAGE_NAME,
            "container_image_id": image_id,
            "overwrite": overwrite,
        },
        dev=dev,
        modules_dir=LIB_SRC_DIR,
        gpu_device=gpu if use_gpu else None,
        env=environment,
        network_disabled=True,
        strict_io_isolation=True,
    )
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run containerized E2FGVI inpainting")
    parser.add_argument("--input-video", "--input_video", dest="input_video", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-video", "--output_video", dest="output_video", required=True)
    parser.add_argument("--metadata-path", "--metadata_path", dest="metadata_path")
    parser.add_argument("--downscale", type=float, default=1.0)
    parser.add_argument("--max-size", "--max_size", dest="max_size", type=int, default=0)
    parser.add_argument(
        "--dilation-iterations",
        "--dilation_iterations",
        dest="dilation_iterations",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dilation-kernel", "--dilation_kernel", dest="dilation_kernel", type=int, default=3
    )
    parser.add_argument(
        "--neighbor-stride", "--neighbor_stride", dest="neighbor_stride", type=int, default=5
    )
    parser.add_argument("--ref-stride", "--ref_stride", dest="ref_stride", type=int, default=20)
    parser.add_argument("--num-ref", "--num_ref", dest="num_ref", type=int, default=-1)
    parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0"], default="cuda:0")
    parser.add_argument("--gpu", type=int, default=0, help="Physical host GPU index")
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-only", "--validate_only", dest="validate_only", action="store_true")
    parser.add_argument(
        "--enrich-completed-metadata",
        "--enrich_completed_metadata",
        dest="enrich_completed_metadata",
        action="store_true",
        help="safely fingerprint an existing legacy completed output/sidecar",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="CPU-only host validation; does not invoke Docker",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dev", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_inpaint(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
