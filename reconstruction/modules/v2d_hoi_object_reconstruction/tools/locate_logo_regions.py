#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Propose logo or trademark regions in one texture image for human review.

The model output is advisory. This tool can write a candidate redaction config,
but it never edits an image or mesh. Review and correct every proposed region
before passing that config to ``redact_texture_regions.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_VLLM_URL = "http://localhost:8000/v1"
NORMALIZED_EXTENT = 1000.0


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("model response does not contain a JSON object")
        candidate = text[start : end + 1]

    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("model response is not a JSON object")
    return parsed


def _pixel_coordinate(value: float, extent: int) -> int:
    if extent <= 0:
        raise ValueError("image dimensions must be positive")
    return min(
        extent - 1,
        int(value * (extent - 1) / NORMALIZED_EXTENT + 0.5),
    )


def _validated_regions(parsed: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    raw_regions = parsed.get("regions", [])
    if not isinstance(raw_regions, list):
        raise ValueError("model response regions must be a list")

    regions: list[dict[str, Any]] = []
    for index, region in enumerate(raw_regions):
        if not isinstance(region, dict):
            raise ValueError(f"model response region {index} must be an object")
        bbox = region.get("bbox")
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox
            )
        ):
            raise ValueError(f"model response region {index} bbox must contain four numbers")

        x1, y1, x2, y2 = (max(0.0, min(NORMALIZED_EXTENT, float(value))) for value in bbox)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"model response region {index} bbox has no area")
        pixel_bbox = [
            _pixel_coordinate(x1, width),
            _pixel_coordinate(y1, height),
            _pixel_coordinate(x2, width),
            _pixel_coordinate(y2, height),
        ]
        if pixel_bbox[2] <= pixel_bbox[0] or pixel_bbox[3] <= pixel_bbox[1]:
            raise ValueError(f"model response region {index} bbox is smaller than one pixel")
        regions.append(
            {
                "label": str(region.get("label", f"region_{index + 1}")),
                "bbox_normalized_1000": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                ],
                "bbox_pixels": pixel_bbox,
            }
        )
    return regions


def _region_name(label: str, index: int) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"{normalized or 'mark'}_{index:02d}"


def _candidate_config(regions: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
    configured = []
    for index, region in enumerate(regions, start=1):
        left, top, right, bottom = region["bbox_pixels"]
        radius = max(3, round(max(right - left, bottom - top) * 0.35))
        configured.append(
            {
                "name": _region_name(region["label"], index),
                "shape": "rectangle",
                "top_left": [left, top],
                "bottom_right": [right, bottom],
                "operation": "blur",
                "radius_px": radius,
            }
        )
    return {
        "schema_version": 1,
        "expected_size": [width, height],
        "regions": configured,
    }


def _tile_bounds(length: int, count: int, overlap_fraction: float) -> list[tuple[int, int]]:
    if length <= 0 or count <= 0:
        raise ValueError("image dimensions and tile counts must be positive")
    if not 0.0 <= overlap_fraction < 0.5:
        raise ValueError("tile overlap must satisfy 0 <= overlap < 0.5")
    bounds = []
    for index in range(count):
        core_start = round(index * length / count)
        core_end = round((index + 1) * length / count)
        overlap = round((core_end - core_start) * overlap_fraction)
        bounds.append((max(0, core_start - overlap), min(length, core_end + overlap)))
    return bounds


def _iou(first: list[int], second: list[int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if not intersection:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def _deduplicate(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(
        regions, key=lambda item: (item["bbox_pixels"][1], item["bbox_pixels"][0])
    ):
        duplicate = next(
            (
                existing
                for existing in kept
                if _iou(candidate["bbox_pixels"], existing["bbox_pixels"]) >= 0.35
            ),
            None,
        )
        if duplicate is None:
            kept.append(candidate)
        else:
            duplicate["bbox_pixels"] = [
                min(duplicate["bbox_pixels"][0], candidate["bbox_pixels"][0]),
                min(duplicate["bbox_pixels"][1], candidate["bbox_pixels"][1]),
                max(duplicate["bbox_pixels"][2], candidate["bbox_pixels"][2]),
                max(duplicate["bbox_pixels"][3], candidate["bbox_pixels"][3]),
            ]
            duplicate["bbox_normalized_1000"] = [
                min(
                    duplicate["bbox_normalized_1000"][0],
                    candidate["bbox_normalized_1000"][0],
                ),
                min(
                    duplicate["bbox_normalized_1000"][1],
                    candidate["bbox_normalized_1000"][1],
                ),
                max(
                    duplicate["bbox_normalized_1000"][2],
                    candidate["bbox_normalized_1000"][2],
                ),
                max(
                    duplicate["bbox_normalized_1000"][3],
                    candidate["bbox_normalized_1000"][3],
                ),
            ]
            duplicate["proposed_by"].extend(candidate["proposed_by"])
    for region in kept:
        region["proposed_by"] = sorted(set(region["proposed_by"]))
    return sorted(kept, key=lambda item: (item["bbox_pixels"][1], item["bbox_pixels"][0]))


def _map_to_atlas(
    region: dict[str, Any],
    bounds: tuple[int, int, int, int],
    atlas_size: tuple[int, int],
    source_pass: str,
) -> dict[str, Any]:
    left, top, _, _ = bounds
    atlas_width, atlas_height = atlas_size
    local_left, local_top, local_right, local_bottom = region["bbox_pixels"]
    pixel_bbox = [
        left + local_left,
        top + local_top,
        left + local_right,
        top + local_bottom,
    ]
    normalized = [
        round(pixel_bbox[0] * NORMALIZED_EXTENT / (atlas_width - 1), 2),
        round(pixel_bbox[1] * NORMALIZED_EXTENT / (atlas_height - 1), 2),
        round(pixel_bbox[2] * NORMALIZED_EXTENT / (atlas_width - 1), 2),
        round(pixel_bbox[3] * NORMALIZED_EXTENT / (atlas_height - 1), 2),
    ]
    return {
        "label": region["label"],
        "bbox_normalized_1000": normalized,
        "bbox_pixels": pixel_bbox,
        "proposed_by": [source_pass],
    }


def _prompt(expected_marks: list[str], *, is_tile: bool) -> str:
    marks = "; ".join(expected_marks)
    image_description = (
        "one high-resolution overlapping tile from a UV texture atlas"
        if is_tile
        else "this full UV texture atlas"
    )
    return f"""Inspect {image_description} and locate every visible occurrence of these expected marks: {marks}.

Search all large and small texture islands, including rotated or repeated appearances. Include only an identifiable logo, wordmark, or trademark symbol matching the expected marks. Do not include barcodes, product instructions, unrelated descriptive text, handwritten notes, or generic shapes. Omit fragments that are too blurred or incomplete to recognize.

Return only strict JSON in this form:
{{"recognizable": true, "regions": [{{"label": "exact mark", "bbox": [x1,y1,x2,y2]}}], "notes": "short"}}

Coordinates must be normalized to the supplied image: left=0, top=0, right=1000, bottom=1000. Use tight boxes around only the mark. If none is recognizable, return recognizable=false and an empty regions list."""


def _parse_tile_grid(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"([1-9][0-9]*)x([1-9][0-9]*)", value.lower())
    if not match:
        raise argparse.ArgumentTypeError("tile grid must use COLUMNSxROWS, for example 2x2")
    return int(match.group(1)), int(match.group(2))


def locate_image(
    image_path: str | Path,
    expected_marks: list[str],
    model: Any,
    *,
    model_name: str,
    backend: str,
    max_new_tokens: int = 700,
    tile_grid: tuple[int, int] | None = None,
    tile_overlap: float = 0.12,
    include_whole_image: bool = True,
) -> dict[str, Any]:
    """Run one advisory location request through a compatible model wrapper."""

    if not expected_marks or any(not mark.strip() for mark in expected_marks):
        raise ValueError("expected_marks must contain at least one non-empty mark")
    source = Path(image_path).expanduser().resolve()
    with Image.open(source) as opened:
        opened.load()
        frame = opened.convert("RGB")
        width, height = frame.size
    if width < 2 or height < 2:
        raise ValueError("input image must be at least 2x2 pixels")

    scan_inputs: list[tuple[str, tuple[int, int, int, int], Image.Image]] = []
    if include_whole_image:
        scan_inputs.append(("whole_image", (0, 0, width, height), frame))
    if tile_grid is not None:
        columns, rows = tile_grid
        x_bounds = _tile_bounds(width, columns, tile_overlap)
        y_bounds = _tile_bounds(height, rows, tile_overlap)
        for row, (top, bottom) in enumerate(y_bounds, start=1):
            for column, (left, right) in enumerate(x_bounds, start=1):
                scan_inputs.append(
                    (
                        f"tile_r{row}_c{column}",
                        (left, top, right, bottom),
                        frame.crop((left, top, right, bottom)),
                    )
                )
    if not scan_inputs:
        raise ValueError("at least one whole-image or tiled scan must be enabled")

    proposed: list[dict[str, Any]] = []
    passes: list[dict[str, Any]] = []
    for name, bounds, scan_image in scan_inputs:
        response = model.generate_from_frames(
            frames=[scan_image],
            prompt=_prompt(expected_marks, is_tile=name != "whole_image"),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        try:
            parsed = _extract_json(response)
            if not isinstance(parsed.get("recognizable"), bool):
                raise ValueError("model response recognizable must be a boolean")
            local_regions = _validated_regions(parsed, *scan_image.size)
            if parsed["recognizable"] != bool(local_regions):
                raise ValueError("model response recognizable flag and regions disagree")
            parse_error = None
        except (ValueError, json.JSONDecodeError) as exc:
            parsed = {}
            local_regions = []
            parse_error = str(exc)
        proposed.extend(
            _map_to_atlas(region, bounds, (width, height), name) for region in local_regions
        )
        passes.append(
            {
                "name": name,
                "atlas_bounds_pixels": list(bounds),
                "regions_proposed": len(local_regions),
                "notes": str(parsed.get("notes", "")),
                "parse_error": parse_error,
                "raw_response": response,
            }
        )

    effective_regions = _deduplicate(proposed)
    recognizable = bool(effective_regions)
    failed_passes = [item["name"] for item in passes if item["parse_error"]]
    scan_status = "incomplete" if failed_passes else "complete"
    return {
        "schema_version": 1,
        "advisory_only": True,
        "input": str(source),
        "input_sha256": _file_hash(source),
        "image_size": [width, height],
        "expected_marks": expected_marks,
        "model": model_name,
        "model_backend": backend,
        "coordinate_system": "full_atlas_pixels_and_normalized_0_to_1000",
        "scan": {
            "whole_image": include_whole_image,
            "tile_grid": list(tile_grid) if tile_grid else None,
            "tile_overlap": tile_overlap if tile_grid else None,
        },
        "scan_status": scan_status,
        "failed_passes": failed_passes,
        "recognizable_proposal": recognizable,
        "regions": effective_regions,
        "passes": passes,
        "candidate_redaction_config": (
            _candidate_config(effective_regions, width, height)
            if scan_status == "complete" and effective_regions
            else None
        ),
    }


def _write_text_atomic(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.write-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        staged.write_text(content, encoding="utf-8")
        os.replace(staged, destination)


def _write_review_overlay(
    image_path: Path, regions: list[dict[str, Any]], destination: Path
) -> None:
    if destination.suffix.lower() != ".png":
        raise ValueError("review overlay output must use a .png extension")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(max(image.size) / 1000))
    for region in regions:
        left, top, right, bottom = region["bbox_pixels"]
        draw.rectangle((left, top, right, bottom), outline=(255, 215, 0), width=line_width)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.write-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        image.save(staged, format="PNG")
        os.replace(staged, destination)


def _validate_output_paths(
    input_image: Path, outputs: list[Path], *, overwrite: bool
) -> list[Path]:
    source = input_image.expanduser().resolve()
    resolved = [path.expanduser().resolve() for path in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output, config, and overlay paths must be different")
    if source in resolved:
        raise ValueError("an output path must not overwrite the input texture")
    existing = [path for path in resolved if path.exists()]
    directories = [path for path in existing if path.is_dir()]
    if directories:
        raise IsADirectoryError(f"output path is a directory: {directories[0]}")
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}; pass --force")
    return resolved


def _model_manager() -> Any:
    repository_root = Path(__file__).resolve().parents[4]
    package_source = repository_root / "video_ingestion_agent" / "src"
    package_source_text = str(package_source)
    if package_source_text not in sys.path:
        sys.path.insert(0, package_source_text)

    from video_ingestion_agent.models.model_manager import get_model_manager

    return get_model_manager()


def _load_model(args: argparse.Namespace) -> Any:
    manager = _model_manager()
    if args.backend == "local":
        return manager.get_model(
            model_name=args.model,
            backend="local",
            device=args.device,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            mm_max_pixels=args.max_pixels,
        )
    return manager.get_model(
        model_name=args.model,
        backend="vllm",
        api_url=args.vllm_url,
        use_local_media=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_image", type=Path)
    parser.add_argument("--expected-mark", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--overlay-output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--backend", choices=("local", "vllm"), default="local")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--vllm-url", default=DEFAULT_VLLM_URL)
    parser.add_argument("--max-pixels", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--tile-grid", type=_parse_tile_grid, default=(2, 2))
    parser.add_argument("--tile-overlap", type=float, default=0.12)
    parser.add_argument(
        "--whole-image-only",
        action="store_true",
        help="disable tiled fallback and scan only the complete image",
    )
    parser.add_argument("--force", action="store_true", help="replace existing output files")
    args = parser.parse_args()

    if args.output.suffix.lower() != ".json":
        raise ValueError("model record output must use a .json extension")
    if args.config_output and args.config_output.suffix.lower() != ".json":
        raise ValueError("candidate config output must use a .json extension")
    if args.overlay_output and args.overlay_output.suffix.lower() != ".png":
        raise ValueError("review overlay output must use a .png extension")

    requested_outputs = [args.output]
    if args.config_output:
        requested_outputs.append(args.config_output)
    if args.overlay_output:
        requested_outputs.append(args.overlay_output)
    resolved_outputs = _validate_output_paths(
        args.input_image, requested_outputs, overwrite=args.force
    )
    output_path = resolved_outputs[0]
    next_output = 1
    config_output = None
    if args.config_output:
        config_output = resolved_outputs[next_output]
        next_output += 1
    overlay_output = resolved_outputs[next_output] if args.overlay_output else None

    result = locate_image(
        args.input_image,
        args.expected_mark,
        _load_model(args),
        model_name=args.model,
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        tile_grid=None if args.whole_image_only else args.tile_grid,
        tile_overlap=args.tile_overlap,
    )
    _write_text_atomic(output_path, json.dumps(result, indent=2) + "\n")
    if result["scan_status"] != "complete":
        for stale_output in (config_output, overlay_output):
            if stale_output is not None:
                stale_output.unlink(missing_ok=True)
        print(
            "logo scan incomplete; diagnostic record written, but no redaction "
            f"config was produced: {', '.join(result['failed_passes'])}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)
    if config_output:
        if result["candidate_redaction_config"] is None:
            config_output.unlink(missing_ok=True)
        else:
            _write_text_atomic(
                config_output,
                json.dumps(result["candidate_redaction_config"], indent=2) + "\n",
            )
    if overlay_output:
        _write_review_overlay(
            args.input_image.expanduser().resolve(), result["regions"], overlay_output
        )
    print(
        f"wrote {len(result['regions'])} advisory region(s) to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
