#!/usr/bin/env python3
"""Post-process SAM2 mask PNGs for reconstruction workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _component_filter(
    mask: np.ndarray,
    *,
    keep_largest_component: bool,
    min_component_area_px: int,
    min_component_area_frac: float,
) -> np.ndarray:
    if mask.sum() == 0:
        return mask

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    if keep_largest_component:
        keep_labels = np.array([int(np.argmax(areas)) + 1], dtype=np.int32)
    else:
        largest = int(areas.max())
        threshold = max(int(min_component_area_px), int(largest * float(min_component_area_frac)))
        keep_labels = np.where(areas >= threshold)[0] + 1

    cleaned = np.zeros_like(mask)
    if keep_labels.size:
        cleaned[np.isin(labels, keep_labels)] = 1
    return cleaned


def _morph(mask: np.ndarray, *, open_px: int, erode_px: int) -> np.ndarray:
    if mask.sum() == 0:
        return mask

    cleaned = mask
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k)
    if erode_px > 0 and cleaned.sum() > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        cleaned = cv2.erode(cleaned, k, iterations=1)
    return cleaned


def _mask_groups(input_dir: Path) -> list[tuple[Path, Path]]:
    if any(input_dir.glob("*.png")):
        return [(input_dir, Path("."))]

    groups = []
    for child in sorted(input_dir.iterdir()):
        if child.is_dir() and any(child.glob("*.png")):
            groups.append((child, child.relative_to(input_dir)))
    return groups


def postprocess_masks(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    keep_largest_component: bool = False,
    min_component_area_px: int = 2000,
    min_component_area_frac: float = 0.01,
    open_px: int = 0,
    erode_px: int = 3,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    groups = _mask_groups(input_dir)
    if not groups:
        raise FileNotFoundError(f"no mask PNGs found under: {input_dir}")

    frame_stats = []
    for src_group, rel_group in groups:
        dst_group = output_dir / rel_group
        dst_group.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_group.glob("*.png")):
            img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"failed to read mask: {src}")

            mask = (img > 127).astype(np.uint8)
            before = int(mask.sum())
            cleaned = _component_filter(
                mask,
                keep_largest_component=keep_largest_component,
                min_component_area_px=min_component_area_px,
                min_component_area_frac=min_component_area_frac,
            )
            cleaned = _morph(cleaned, open_px=open_px, erode_px=erode_px)
            after = int(cleaned.sum())

            cv2.imwrite(str(dst_group / src.name), cleaned.astype(np.uint8) * 255)
            frame_stats.append(
                {
                    "group": str(rel_group),
                    "frame": src.name,
                    "area_before": before,
                    "area_after": after,
                    "area_ratio": float(after / before) if before else 0.0,
                }
            )

    before_areas = np.array([s["area_before"] for s in frame_stats], dtype=np.int64)
    after_areas = np.array([s["area_after"] for s in frame_stats], dtype=np.int64)
    ratios = np.array([s["area_ratio"] for s in frame_stats if s["area_before"] > 0], dtype=np.float64)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "frames": len(frame_stats),
        "keep_largest_component": bool(keep_largest_component),
        "min_component_area_px": int(min_component_area_px),
        "min_component_area_frac": float(min_component_area_frac),
        "open_px": int(open_px),
        "erode_px": int(erode_px),
        "area_before_total": int(before_areas.sum()),
        "area_after_total": int(after_areas.sum()),
        "area_ratio_mean": float(ratios.mean()) if ratios.size else None,
        "area_ratio_median": float(np.median(ratios)) if ratios.size else None,
        "area_ratio_p05": float(np.percentile(ratios, 5)) if ratios.size else None,
        "area_ratio_p95": float(np.percentile(ratios, 95)) if ratios.size else None,
        "empty_after_frames": int(np.sum(after_areas == 0)),
    }
    (output_dir / "postprocess_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process SAM2 mask PNGs")
    parser.add_argument("--input_dir", required=True, help="Input mask root, e.g. masks_raw_sam2")
    parser.add_argument("--output_dir", required=True, help="Output mask root, e.g. masks")
    parser.add_argument("--keep_largest_component", action="store_true")
    parser.add_argument("--min_component_area_px", type=int, default=2000)
    parser.add_argument("--min_component_area_frac", type=float, default=0.01)
    parser.add_argument("--open_px", type=int, default=0)
    parser.add_argument("--erode_px", type=int, default=3)
    args = parser.parse_args()

    summary = postprocess_masks(
        args.input_dir,
        args.output_dir,
        keep_largest_component=args.keep_largest_component,
        min_component_area_px=args.min_component_area_px,
        min_component_area_frac=args.min_component_area_frac,
        open_px=args.open_px,
        erode_px=args.erode_px,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
