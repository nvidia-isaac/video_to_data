# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point for validation and E2FGVI inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .validation import (
    InferenceConfig,
    InputValidationError,
    build_metadata,
    canonical_json,
    enrich_completed_metadata,
    validate_output_paths,
    validate_run,
    write_metadata,
)


def _default_metadata_path(output_video: str) -> str:
    return f"{output_video}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inpaint a video using E2FGVI-HQ")
    parser.add_argument("--input-video", "--input_video", dest="input_video", required=True)
    parser.add_argument("--masks", required=True, help="Boolean .npy array shaped [frames, height, width]")
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--validate-only",
        "--validate_only",
        dest="validate_only",
        action="store_true",
        help="Validate and write metadata without importing the model or requiring a GPU",
    )
    parser.add_argument(
        "--enrich-completed-metadata",
        "--enrich_completed_metadata",
        dest="enrich_completed_metadata",
        action="store_true",
        help=(
            "validate a legacy completed sidecar and atomically add the current "
            "output fingerprint; does not run inference"
        ),
    )
    parser.add_argument("--container-image", "--container_image", dest="container_image")
    parser.add_argument(
        "--container-image-id", "--container_image_id", dest="container_image_id"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata_path = args.metadata_path or _default_metadata_path(args.output_video)
    output = Path(args.output_video).resolve()
    metadata = Path(metadata_path).resolve()
    validate_output_paths(
        args.input_video,
        args.masks,
        args.checkpoint,
        output,
        metadata,
    )
    if args.validate_only and args.enrich_completed_metadata:
        raise InputValidationError(
            "--validate-only and --enrich-completed-metadata are mutually exclusive"
        )
    if args.enrich_completed_metadata:
        if args.container_image is not None or args.container_image_id is not None:
            raise InputValidationError(
                "legacy enrichment cannot retroactively assert a container image identity"
            )
        report = enrich_completed_metadata(
            metadata,
            args.input_video,
            args.masks,
            args.checkpoint,
            output,
        )
        print(canonical_json(report), end="")
        return 0
    if not args.overwrite:
        targets = [metadata] if args.validate_only else [output, metadata]
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite existing output(s): {', '.join(existing)}")

    config = InferenceConfig(
        downscale=args.downscale,
        max_size=args.max_size,
        dilation_iterations=args.dilation_iterations,
        dilation_kernel=args.dilation_kernel,
        neighbor_stride=args.neighbor_stride,
        ref_stride=args.ref_stride,
        num_ref=args.num_ref,
        device=args.device,
        codec=args.codec,
        seed=args.seed,
    )
    plan = validate_run(args.input_video, args.masks, args.checkpoint, config)
    if args.validate_only:
        report = build_metadata(
            plan,
            args.input_video,
            args.masks,
            args.checkpoint,
            args.output_video,
            "validated",
            container_image=args.container_image,
            container_image_id=args.container_image_id,
        )
        write_metadata(metadata, report)
        print(canonical_json(report), end="")
        return 0

    committing = build_metadata(
        plan,
        args.input_video,
        args.masks,
        args.checkpoint,
        args.output_video,
        "committing",
        container_image=args.container_image,
        container_image_id=args.container_image_id,
    )
    # Invalidate any prior completed sidecar before inference can replace its
    # output.  A crash from this point onward therefore leaves a non-complete
    # commit marker instead of falsely resuming a new or half-published video.
    write_metadata(metadata, committing)

    from .inference import run_inference

    run_inference(
        plan,
        args.input_video,
        args.masks,
        args.checkpoint,
        args.output_video,
    )
    report = build_metadata(
        plan,
        args.input_video,
        args.masks,
        args.checkpoint,
        args.output_video,
        "completed",
        container_image=args.container_image,
        container_image_id=args.container_image_id,
    )
    if report["inputs"] != committing["inputs"]:
        raise InputValidationError(
            "an E2FGVI input changed during inference; completed metadata was not committed"
        )
    write_metadata(metadata, report)
    print(canonical_json(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
