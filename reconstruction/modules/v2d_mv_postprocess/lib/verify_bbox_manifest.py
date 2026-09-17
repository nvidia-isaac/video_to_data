"""Verify a labeled-bbox directory against its orchestration content hash."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from v2d.common.video import FrameSource


def logical_bbox_manifest(directory: str | Path) -> tuple[list[dict], str]:
    root = Path(directory)
    entries = []
    for path in sorted(root.glob("*.json"), key=lambda item: item.name):
        payload = path.read_bytes()
        entries.append({
            "name": path.name,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
    if not entries:
        raise ValueError(f"No labeled bbox JSON files found in {root}")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return entries, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_bbox_manifest(
    directory: str | Path,
    expected_sha256: str,
    expected_manifest: list[dict] | None = None,
) -> list[dict]:
    entries, actual = logical_bbox_manifest(directory)
    if expected_manifest is None:
        if actual == expected_sha256:
            return entries
        raise ValueError(
            "Labeled bbox content changed after reconstruction submission: "
            f"expected {expected_sha256}, found {actual}"
        )

    canonical_manifest = json.dumps(
        expected_manifest, sort_keys=True, separators=(",", ":")
    )
    manifest_sha256 = hashlib.sha256(canonical_manifest.encode("utf-8")).hexdigest()
    if manifest_sha256 != expected_sha256:
        raise ValueError(
            "Stored labeled bbox manifest does not match its expected identity: "
            f"expected {expected_sha256}, found {manifest_sha256}"
        )
    expected_logical = [
        {key: item[key] for key in ("name", "size", "sha256")}
        for item in expected_manifest
    ]
    if entries != expected_logical:
        raise ValueError(
            "Labeled bbox content changed after reconstruction submission: "
            f"expected {expected_logical!r}, found {entries!r}"
        )
    return entries


def _best_detection(path: Path) -> tuple[int, dict, str]:
    payload = json.loads(path.read_text())
    candidates = [
        (int(stem), detection)
        for stem, detections in payload.items()
        for detection in detections
    ]
    if not candidates:
        raise ValueError(f"No bbox detections found in {path}")
    frame, detection = max(
        candidates, key=lambda item: float(item[1].get("confidence", 0.0))
    )
    box = detection.get("box") or {}
    required = ("x0", "y0", "x1", "y1")
    if any(key not in box for key in required):
        raise ValueError(f"Malformed bbox in {path}")
    return frame, box, str(detection.get("label", "object"))


def render_bbox_previews(
    bbox_dir: str | Path, rgb_dir: str | Path, preview_dir: str | Path,
) -> list[dict]:
    """Validate prompt coordinates/stems against decoded frames and render previews."""
    bbox_dir, rgb_dir, preview_dir = map(Path, (bbox_dir, rgb_dir, preview_dir))
    records = []
    for bbox_path in sorted(bbox_dir.glob("*.json")):
        camera = bbox_path.stem
        source_path = rgb_dir / f"{camera}.h5"
        if not source_path.is_file():
            raise ValueError(f"No RGB HDF5 matches bbox camera {camera}")
        frame_index, box, label = _best_detection(bbox_path)
        with FrameSource.from_path(source_path) as source:
            if frame_index < 0 or frame_index >= source.n_frames:
                raise ValueError(
                    f"Bbox stem {frame_index} is outside {camera}'s {source.n_frames} frames"
                )
            frame = np.asarray(source[frame_index])
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = (float(box[key]) for key in ("x0", "y0", "x1", "y1"))
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(
                f"Bbox coordinates for {camera} do not match decoded frame {width}x{height}"
            )
        image = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(image)
        green = (0, 255, 0)
        line_width = max(2, round(min(width, height) / 300))
        draw.rectangle((x0, y0, x1, y1), outline=green, width=line_width)
        text_box = draw.textbbox((0, 0), label, stroke_width=1)
        label_y = max(0, round(y0) - (text_box[3] - text_box[1]) - 3)
        draw.text(
            (max(0, round(x0)), label_y), label, fill=green,
            stroke_width=1, stroke_fill=(0, 0, 0),
        )
        destination = preview_dir / f"{camera}_bbox.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        records.append({
            "camera": camera, "frame_index": frame_index,
            "frame_width": width, "frame_height": height,
            "preview": destination.name,
        })
    if not records:
        raise ValueError("No labeled bbox previews were produced")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox-dir", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--expected-manifest-json",
        help="Manifest JSON containing names, sizes, ETags, and per-file hashes",
    )
    parser.add_argument("--rgb-dir")
    parser.add_argument("--preview-dir")
    args = parser.parse_args()
    expected_manifest = None
    if args.expected_manifest_json:
        expected_manifest = json.loads(Path(args.expected_manifest_json).read_text())
    entries = verify_bbox_manifest(
        args.bbox_dir, args.expected_sha256, expected_manifest
    )
    if bool(args.rgb_dir) != bool(args.preview_dir):
        parser.error("--rgb-dir and --preview-dir must be provided together")
    previews = (
        render_bbox_previews(args.bbox_dir, args.rgb_dir, args.preview_dir)
        if args.rgb_dir else []
    )
    print(f"Verified {len(entries)} labeled bbox file(s): {args.expected_sha256}")
    if previews:
        print(f"Rendered {len(previews)} labeled bbox preview(s) to {args.preview_dir}")


if __name__ == "__main__":
    main()
