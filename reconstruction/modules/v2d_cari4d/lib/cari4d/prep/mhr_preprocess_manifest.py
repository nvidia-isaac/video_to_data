from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from prep.mhr_export_utils import frame_names


HUMAN_POSE_VALIDITY_EXPRESSION = "structural_ok & not failure_segments(any human/object) & not excluded_source_frame_ranges"
OBJECT_POSE_VALIDITY_EXPRESSION = "structural_ok & pose_valid_mask(if present; missing=true) & not failure_segments(any human/object) & not excluded_source_frame_ranges"
FRAME_VALIDITY_EXPRESSION = "human_pose_valid_mask & object_pose_valid_mask"


def _frame_name(index: int) -> str:
    return f"{int(index):06d}"


def compute_pose_validity_masks(frame_count: int, object_pose_valid_mask: np.ndarray | None, failure_segments: Sequence[Mapping[str, Any]], *, source_frame_indices: np.ndarray | None = None, excluded_source_frame_ranges: Sequence[Mapping[str, Any]] = ()) -> dict[str, np.ndarray]:
    frame_count = int(frame_count)
    source_indices = np.arange(frame_count, dtype=np.int64) if source_frame_indices is None else np.asarray(source_frame_indices, dtype=np.int64).reshape(-1)
    if source_indices.shape != (frame_count,) or np.any(source_indices < 0) or np.any(np.diff(source_indices) <= 0):
        raise ValueError(f"source_frame_indices must be a strictly increasing vector with shape {(frame_count,)}, got {source_indices.shape}")
    object_annotation_valid = np.ones(frame_count, dtype=bool) if object_pose_valid_mask is None else np.asarray(object_pose_valid_mask).reshape(-1)
    if object_annotation_valid.shape != (frame_count,) and len(source_indices) and source_indices[-1] < len(object_annotation_valid):
        object_annotation_valid = object_annotation_valid[source_indices]
    if object_annotation_valid.shape != (frame_count,):
        raise ValueError(f"object pose-valid mask from pose_valid_mask.npy has shape {object_annotation_valid.shape}; expected {(frame_count,)}")
    if object_annotation_valid.dtype != np.dtype("bool"):
        if not np.isin(object_annotation_valid, (0, 1)).all():
            raise ValueError("object pose-valid mask from pose_valid_mask.npy contains values other than 0/1")
        object_annotation_valid = object_annotation_valid.astype(bool)
    failure_invalid = np.zeros(frame_count, dtype=bool)
    for segment in failure_segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if start < 0 or end < start or (source_frame_indices is None and end > frame_count):
            raise ValueError(f"Invalid end-exclusive failure segment [{start}, {end}) for {frame_count} frames")
        failure_invalid |= (source_indices >= start) & (source_indices < end)
    excluded_invalid = np.zeros(frame_count, dtype=bool)
    for segment in excluded_source_frame_ranges:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if start < 0 or end <= start or (source_frame_indices is None and end > frame_count):
            raise ValueError(f"Invalid end-exclusive excluded source-frame range [{start}, {end}) for {frame_count} frames")
        excluded_invalid |= (source_indices >= start) & (source_indices < end)
    human_pose_valid = ~(failure_invalid | excluded_invalid)
    object_pose_valid = object_annotation_valid & ~failure_invalid & ~excluded_invalid
    frame_valid = human_pose_valid & object_pose_valid
    return {"human_pose_valid_mask": human_pose_valid, "object_pose_valid_mask": object_pose_valid, "frame_valid_mask": frame_valid, "object_pose_annotation_valid_mask": object_annotation_valid, "failure_invalid_mask": failure_invalid, "excluded_source_frame_invalid_mask": excluded_invalid}


def compute_validity_manifest(seq_name: str, frame_count: int, object_pose_valid_mask: np.ndarray | None, failure_segments: Sequence[Mapping[str, Any]], *, frames: Sequence[str] | None = None, source_frame_indices: np.ndarray | None = None, excluded_source_frame_ranges: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    frame_count = int(frame_count)
    frame_values = [_frame_name(index) for index in range(frame_count)] if frames is None else [str(frame) for frame in frames]
    if len(frame_values) != frame_count:
        raise ValueError(f"frames has {len(frame_values)} entries; expected {frame_count}")
    masks = compute_pose_validity_masks(frame_count, object_pose_valid_mask, failure_segments, source_frame_indices=source_frame_indices, excluded_source_frame_ranges=excluded_source_frame_ranges)
    human_pose_valid = masks["human_pose_valid_mask"]
    object_pose_valid = masks["object_pose_valid_mask"]
    valid = masks["frame_valid_mask"]
    object_annotation_valid = masks["object_pose_annotation_valid_mask"]
    failure_invalid = masks["failure_invalid_mask"]
    excluded_invalid = masks["excluded_source_frame_invalid_mask"]
    accepted = [frame_values[index] for index in np.flatnonzero(valid)]
    rejection_reasons = {}
    for index in np.flatnonzero(~valid):
        reasons = []
        if not object_annotation_valid[index]:
            reasons.append("object_pose_valid_mask")
        if failure_invalid[index]:
            reasons.append("failure_segments")
        if excluded_invalid[index]:
            reasons.append("excluded_source_frame_ranges")
        rejection_reasons[frame_values[index]] = reasons
    chunks = []
    start = None
    for index, is_valid in enumerate(np.append(valid, False)):
        if is_valid and start is None:
            start = index
        elif not is_valid and start is not None:
            chunks.append({"start": int(start), "end_exclusive": int(index)})
            start = None
    return {
        "sequence": str(seq_name),
        "frame_count": frame_count,
        "frames": frame_values,
        "human_pose_valid_mask": human_pose_valid.tolist(),
        "object_pose_valid_mask": object_pose_valid.tolist(),
        "frame_valid_mask": valid.tolist(),
        "human_pose_valid_frame_count": int(human_pose_valid.sum()),
        "object_pose_valid_frame_count": int(object_pose_valid.sum()),
        "valid_frame_count": int(valid.sum()),
        "invalid_frame_count": int((~valid).sum()),
        "accepted_frames": accepted,
        "rejected_frames": list(rejection_reasons),
        "rejection_reasons": rejection_reasons,
        "valid_chunks": chunks,
        "failure_segments_end_semantics": "exclusive",
        "pose_valid_mask_scope": "object_pose_only",
        "pose_valid_mask_missing_default": True,
        "failure_segments_scope": "conservative_human_and_object",
        "excluded_source_frame_ranges": [dict(segment) for segment in excluded_source_frame_ranges],
        "excluded_source_frame_ranges_scope": "conservative_human_and_object",
        "human_pose_validity_expression": HUMAN_POSE_VALIDITY_EXPRESSION,
        "object_pose_validity_expression": OBJECT_POSE_VALIDITY_EXPRESSION,
        "validity_expression": FRAME_VALIDITY_EXPRESSION,
    }


def aligned_manifest_pose_validity(base_frames: Sequence[str], manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    base = [str(frame) for frame in base_frames]
    manifest_frames = [str(frame) for frame in manifest["frames"]]
    if manifest_frames != base:
        mismatch = next((index for index, pair in enumerate(zip(manifest_frames, base)) if pair[0] != pair[1]), min(len(manifest_frames), len(base)))
        raise ValueError(f"Validity manifest frames do not match packed GT at index {mismatch}: manifest={manifest_frames[mismatch:mismatch + 3]} packed={base[mismatch:mismatch + 3]}")
    frame_mask = np.asarray(manifest["frame_valid_mask"], dtype=bool)
    human_mask = np.asarray(manifest.get("human_pose_valid_mask", frame_mask), dtype=bool)
    object_mask = np.asarray(manifest.get("object_pose_valid_mask", frame_mask), dtype=bool)
    for key, mask in (("human_pose_valid_mask", human_mask), ("object_pose_valid_mask", object_mask), ("frame_valid_mask", frame_mask)):
        if mask.shape != (len(base),):
            raise ValueError(f"Validity manifest {key} has shape {mask.shape}; expected {(len(base),)}")
    expected_frame_mask = human_mask & object_mask
    if not np.array_equal(frame_mask, expected_frame_mask):
        raise ValueError("Validity manifest frame_valid_mask must equal human_pose_valid_mask & object_pose_valid_mask")
    accepted = [frame for frame, is_valid in zip(base, frame_mask) if is_valid]
    if accepted != [str(frame) for frame in manifest["accepted_frames"]]:
        raise ValueError("Validity manifest accepted_frames does not match frame_valid_mask")
    return {"human_pose_valid_mask": human_mask, "object_pose_valid_mask": object_mask, "frame_valid_mask": frame_mask}


def aligned_manifest_validity(base_frames: Sequence[str], manifest: Mapping[str, Any]) -> np.ndarray:
    return aligned_manifest_pose_validity(base_frames, manifest)["frame_valid_mask"]


def contiguous_frame_starts(frames: Sequence[str], clip_len: int, window: int, include_terminal: bool = False) -> list[int]:
    clip_len = int(clip_len)
    window = int(window)
    if clip_len <= 0 or window <= 0:
        raise ValueError("clip_len and window must be positive")
    values = [str(frame) for frame in frames]
    if not values:
        return []
    def starts_for_chunk(chunk_start: int, chunk_end: int) -> list[int]:
        terminal = chunk_end - clip_len
        if terminal < chunk_start:
            return []
        starts = list(range(chunk_start, terminal + 1, window))
        if include_terminal and starts[-1] != terminal:
            starts.append(terminal)
        return starts
    if not all(Path(value).name.isdigit() for value in values):
        return starts_for_chunk(0, len(values))
    numeric = [int(Path(value).name) for value in values]
    starts = []
    chunk_start = 0
    for index in range(1, len(numeric) + 1):
        if index == len(numeric) or numeric[index] != numeric[index - 1] + 1:
            starts.extend(starts_for_chunk(chunk_start, index))
            chunk_start = index
    return starts


def select_manifest_frames(base_frames: Sequence[str], manifest: Mapping[str, Any]) -> list[str]:
    available = {str(frame) for frame in base_frames}
    accepted = [str(frame) for frame in manifest["accepted_frames"]]
    missing = [frame for frame in accepted if frame not in available]
    if missing:
        raise ValueError(f"Validity manifest contains {len(missing)} frames absent from packed GT; first missing: {missing[:5]}")
    return accepted


def build_validity_manifest(export_seq: str | Path) -> dict[str, Any]:
    export_seq = Path(export_seq)
    poses = np.load(export_seq / "poses.npy", mmap_mode="r")
    frame_count = int(poses.shape[0])
    frames = frame_names(export_seq, 0)
    if len(frames) != frame_count:
        raise ValueError(f"Commercial object-pose timeline has {frame_count} frames but the interaction-trim RGB timeline has {len(frames)}: {export_seq}")
    frame_indices = np.asarray([int(Path(frame).stem) for frame in frames], dtype=np.int64)
    pose_path = export_seq / "pose_valid_mask.npy"
    pose_valid = np.load(pose_path) if pose_path.is_file() else None
    failure_path = export_seq / "failure_segments.json"
    segments = json.loads(failure_path.read_text()) if failure_path.is_file() else []
    trim_path = export_seq / "interaction_trim.json"
    trim = json.loads(trim_path.read_text()) if trim_path.is_file() else {}
    return compute_validity_manifest(export_seq.name, frame_count, pose_valid, segments, frames=frames, source_frame_indices=frame_indices, excluded_source_frame_ranges=trim.get("excluded_source_frame_ranges", []))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cleaned-frame manifest for one Daniel HOI sequence.")
    parser.add_argument("export_seq")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = build_validity_manifest(args.export_seq)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, out)
    print(f"saved validity manifest to {out}")


if __name__ == "__main__":
    main()
