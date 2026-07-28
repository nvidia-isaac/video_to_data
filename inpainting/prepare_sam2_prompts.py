"""Resolve normalized arm boxes into Video2Data SAM2 prompt JSON and preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .video_io import probe_video


def prepare(
    config_path: Path,
    sequence_id: str,
    video_path: Path,
    prompts_output: Path,
    preview_output: Path,
) -> dict:
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "v2d.inpainting.sam2-prompts/v1":
        raise ValueError("Unsupported prompt config schema")
    if sequence_id not in config["sequences"]:
        raise KeyError(f"No arm prompt config for {sequence_id}")
    sequence = config["sequences"][sequence_id]
    geometry = probe_video(video_path)
    frame_index = int(sequence["frame_index"])
    if not 0 <= frame_index < geometry.frame_count:
        raise ValueError(f"Reference frame {frame_index} is outside the source video")
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode reference frame {frame_index}")

    prompts = []
    for object_id, (label, normalized) in enumerate(sequence["boxes"].items(), start=1):
        if len(normalized) != 4 or any(not 0 <= value <= 1 for value in normalized):
            raise ValueError(f"Invalid normalized box for {label}: {normalized}")
        x0, y0, x1, y1 = normalized
        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"Degenerate normalized box for {label}: {normalized}")
        box = {
            "x0": float(round(x0 * (geometry.width - 1))),
            "y0": float(round(y0 * (geometry.height - 1))),
            "x1": float(round(x1 * (geometry.width - 1))),
            "y1": float(round(y1 * (geometry.height - 1))),
        }
        prompts.append(
            {
                "frame_index": frame_index,
                "object_id": object_id,
                "points": None,
                "point_labels": None,
                "box": box,
                "mask_path": None,
            }
        )
        cv2.rectangle(
            frame,
            (int(box["x0"]), int(box["y0"])),
            (int(box["x1"]), int(box["y1"])),
            (50, 210, 255) if object_id == 1 else (70, 90, 255),
            5,
        )
        cv2.putText(
            frame,
            f"id={object_id} {label}",
            (int(box["x0"]) + 8, max(35, int(box["y0"]) + 35)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    result = {
        "prompts": prompts,
        "metadata": {
            "schema_version": config["schema_version"],
            "sequence_id": sequence_id,
            "source_video": str(video_path.resolve()),
            "geometry": geometry.as_dict(),
            "role": "shared_initial_arm_segmentation",
        },
    }
    # Sam2Prompts ignores additional top-level metadata.
    prompts_output.parent.mkdir(parents=True, exist_ok=True)
    prompts_output.write_text(json.dumps(result, indent=2) + "\n")
    preview_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview_output), frame):
        raise RuntimeError(f"Could not write {preview_output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--prompts-output", required=True, type=Path)
    parser.add_argument("--preview-output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(
        args.config,
        args.sequence_id,
        args.video,
        args.prompts_output,
        args.preview_output,
    )
    print(f"Wrote {len(result['prompts'])} arm prompts -> {args.prompts_output.resolve()}")


if __name__ == "__main__":
    main()
