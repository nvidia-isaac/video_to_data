"""Union per-object SAM2 PNGs into the strict arm-mask contract and preview."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .contracts import ContractError
from .video_io import probe_video


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Drop isolated SAM2 speckles while preserving the arm-shaped component."""
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        return mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def convert(
    video_path: Path,
    masks_dir: Path,
    output_mask: Path,
    preview_video: Path,
    metadata_path: Path | None = None,
    max_coverage: float = 0.65,
    keep_largest_components: bool = True,
) -> dict:
    geometry = probe_video(video_path)
    object_dirs = sorted(
        path for path in masks_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )
    if not object_dirs:
        raise FileNotFoundError(f"No numeric SAM2 object directories under {masks_dir}")
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    temporary_mask = output_mask.with_name(f"{output_mask.stem}.partial.npy")
    union_masks = np.lib.format.open_memmap(
        temporary_mask,
        mode="w+",
        dtype=np.bool_,
        shape=(geometry.frame_count, geometry.height, geometry.width),
    )
    coverage = np.zeros(geometry.frame_count, dtype=np.float64)
    for frame_index in range(geometry.frame_count):
        union = np.zeros((geometry.height, geometry.width), dtype=np.bool_)
        for object_dir in object_dirs:
            path = object_dir / f"{frame_index:06d}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Missing SAM2 mask: {path}")
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != union.shape:
                raise ContractError(
                    f"SAM2 mask {path} must be {union.shape}, got {None if mask is None else mask.shape}"
                )
            object_mask = mask != 0
            if keep_largest_components:
                object_mask = _largest_connected_component(object_mask)
            union |= object_mask
        coverage[frame_index] = float(union.mean())
        if coverage[frame_index] <= 0:
            raise ContractError(f"Union arm mask is empty at frame {frame_index}")
        if coverage[frame_index] > max_coverage:
            raise ContractError(
                f"Union arm mask covers {coverage[frame_index]:.1%} at frame {frame_index}; "
                f"limit is {max_coverage:.1%}"
            )
        union_masks[frame_index] = union
    union_masks.flush()
    del union_masks
    temporary_mask.replace(output_mask)

    capture = cv2.VideoCapture(str(video_path))
    preview_video.parent.mkdir(parents=True, exist_ok=True)
    temporary_preview = preview_video.with_name(
        f"{preview_video.stem}.partial{preview_video.suffix}"
    )
    writer = cv2.VideoWriter(
        str(temporary_preview),
        cv2.VideoWriter_fourcc(*"mp4v"),
        geometry.fps,
        (geometry.width, geometry.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {temporary_preview}")
    masks = np.load(output_mask, mmap_mode="r")
    try:
        for frame_index in range(geometry.frame_count):
            ok, frame = capture.read()
            if not ok:
                raise ContractError(f"Video decode failed at frame {frame_index}")
            mask = masks[frame_index]
            overlay = np.full_like(frame, (180, 40, 220))
            frame[mask] = cv2.addWeighted(frame, 0.35, overlay, 0.65, 0)[mask]
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"shared SAM2 arm mask | frame {frame_index:04d}",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    temporary_preview.replace(preview_video)

    metadata = {
        "schema_version": "v2d.inpainting.arm-mask-run/v1",
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "role": "shared_initial_arm_segmentation",
        "source_video": str(video_path.resolve()),
        "source_sam2_masks": str(masks_dir.resolve()),
        "object_ids": [int(path.name) for path in object_dirs],
        "component_filter": (
            "largest_connected_component_per_object"
            if keep_largest_components
            else "none"
        ),
        "output_mask": str(output_mask.resolve()),
        "preview_video": str(preview_video.resolve()),
        "geometry": geometry.as_dict(),
        "coverage": {
            "min": float(coverage.min()),
            "median": float(np.median(coverage)),
            "max": float(coverage.max()),
        },
    }
    if metadata_path is None:
        metadata_path = output_mask.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--masks-dir", required=True, type=Path)
    parser.add_argument("--output-mask", required=True, type=Path)
    parser.add_argument("--preview-video", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--max-coverage", type=float, default=0.65)
    parser.add_argument(
        "--keep-all-components",
        action="store_true",
        help="Disable the default per-object isolated-component cleanup.",
    )
    args = parser.parse_args()
    metadata = convert(
        args.video,
        args.masks_dir,
        args.output_mask,
        args.preview_video,
        args.metadata,
        args.max_coverage,
        not args.keep_all_components,
    )
    print(
        f"Wrote {metadata['geometry']['frame_count']} union masks -> "
        f"{metadata['output_mask']} (median coverage {metadata['coverage']['median']:.1%})"
    )


if __name__ == "__main__":
    main()
