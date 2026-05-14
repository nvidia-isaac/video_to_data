# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Segment loaded/retargeted sequences into atomic hand-object interaction clips.

ONLY TESTED ON HOT3D so far. The code uses generic ManoSharpaData fields
(``mano_*_tips_distance``, ``mano_*_object_contact_part_ids``, etc.) so it
should work on other datasets, but the contact thresholds, gap-bridging
windows, and per-body majority-vote heuristics have only been validated on
HOT3D's 30 Hz, multi-object kitchen sequences. Use on taco / arctic / oakink2
will require re-tuning ``--threshold``, ``--gap_frames``, ``--min_segment_len``.

Reads from a ManoSharpaData parquet dir and writes sliced parquets to
``<input>_segmented/`` (or ``--output_dir``) plus a ``segment_manifest.csv``
sidecar.

Interaction modes
-----------------
A — one hand active, any number of objects
B — both hands active, touching the same object body
C — both hands active, touching different object bodies

Contact detection
-----------------
A hand is considered "active" on frame t when:
    min(mano_{side}_tips_distance[t]) < threshold

Object assignment per segment: majority vote of non-zero values in
mano_{side}_object_contact_part_ids across all active frames of that segment.

Grace-period smoothing: short contact gaps < gap_frames are bridged so that
brief lifts don't fragment a single interaction into many tiny clips.

Usage
-----
  python scripts/segment_to_atomic.py \\
      --input_dir ~/datasets/.../hot3d_processed \\
      --dry_run

  python scripts/segment_to_atomic.py \\
      --input_dir ~/datasets/.../taco_processed \\
      --sequence_id some_sequence_id
"""

import argparse
import csv
import logging
import re
import sys
from collections import Counter
from pathlib import Path

# Allow running directly from scripts/ without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent  # robotic_grounding/
_SOURCE_DIR = str(_REPO_ROOT / "source" / "robotic_grounding")
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)

import numpy as np
from robotic_grounding.retarget.data_logger import (
    MANO_FIELDS,
    OBJECT_FIELDS,
    SHARPA_FIELDS,
    ManoSharpaData,
    add_sequence_filter_args,
    filter_sequence_ids,
    list_sequence_ids,
)

# Defined in retarget_utils.py but importing that module pulls in torch/pinocchio.
DEFAULT_PARTITION_COLS = ["sequence_id", "robot_name"]
from tqdm import tqdm

logging.getLogger().setLevel(logging.ERROR)

# All time-series field names — used when slicing a sequence by frame range.
_TIMESERIES_FIELDS: list[str] = [
    name
    for (name, _, _, is_ts) in (MANO_FIELDS + SHARPA_FIELDS + OBJECT_FIELDS)
    if is_ts
]

SEGMENT_STILL_PADDING_FRAMES = 10  # still frames prepended/appended to each segment

MANIFEST_COLUMNS = [
    "segment_id",
    "parent_sequence_id",
    "segment_idx",
    "mode",
    "start_frame",
    "end_frame",
    "num_frames",
    "duration_s",
    "object_name",
]


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------


def detect_contact(tips: np.ndarray, threshold: float) -> np.ndarray:
    """Return bool[T] — True where the hand is within threshold of any object surface."""
    return tips.min(axis=1) < threshold


def fill_gaps(mask: np.ndarray, gap_frames: int) -> np.ndarray:
    """Fill runs of False shorter than gap_frames inside a True region.

    Equivalent to binary closing with a flat window of size gap_frames.
    """
    if gap_frames <= 0:
        return mask
    result = mask.copy()
    T = len(mask)
    i = 0
    while i < T:
        if not result[i]:
            # Find end of False run
            j = i
            while j < T and not result[j]:
                j += 1
            gap_len = j - i
            # Close the gap only if it is surrounded by True on both sides
            if gap_len < gap_frames and i > 0 and j < T:
                result[i:j] = True
            i = j
        else:
            i += 1
    return result


def get_dominant_body(part_ids: np.ndarray, active_mask: np.ndarray) -> int | None:
    """Majority vote of non-zero contact body IDs on active frames."""
    active_ids = part_ids[active_mask].ravel()
    nonzero = active_ids[active_ids != 0]
    if len(nonzero) == 0:
        return None
    return int(Counter(nonzero.tolist()).most_common(1)[0][0])


def find_active_windows(
    right_active: np.ndarray, left_active: np.ndarray
) -> list[tuple[int, int]]:
    """Return list of (start, end) frame indices where at least one hand is active.

    end is exclusive (Python slice convention).
    """
    combined = right_active | left_active
    windows: list[tuple[int, int]] = []
    T = len(combined)
    i = 0
    while i < T:
        if combined[i]:
            j = i
            while j < T and combined[j]:
                j += 1
            windows.append((i, j))
            i = j
        else:
            i += 1
    return windows


def _per_frame_dominant_body(cpid_r: np.ndarray, cpid_l: np.ndarray) -> np.ndarray:
    """Return int[T] — dominant contacted body ID per frame (0 if no contact)."""
    combined = np.concatenate([cpid_r, cpid_l], axis=1)  # [T, 32]
    T = len(combined)
    result = np.zeros(T, dtype=np.int32)
    for t in range(T):
        nonzero = combined[t][combined[t] != 0]
        if len(nonzero):
            result[t] = Counter(nonzero.tolist()).most_common(1)[0][0]
    return result


def split_by_object(
    start: int,
    end: int,
    per_frame_body: np.ndarray,
    gap_frames: int,
) -> list[tuple[int, int]]:
    """Split [start, end) into sub-windows wherever the dominant touched body changes.

    Short zero-gaps (< gap_frames) between same-body runs are bridged.
    Returns a list of (sub_start, sub_end) pairs in global frame coordinates.
    """
    window = per_frame_body[start:end]
    T = len(window)
    sub_windows: list[tuple[int, int]] = []
    i = 0

    while i < T:
        if window[i] == 0:
            i += 1
            continue

        current_body = int(window[i])
        sub_start = i
        j = i + 1

        while j < T:
            if window[j] == current_body:
                j += 1
            elif window[j] == 0:
                # Look ahead: does the same body resume within gap_frames?
                k = j
                while k < T and window[k] == 0:
                    k += 1
                if k < T and int(window[k]) == current_body and (k - j) < gap_frames:
                    j = k + 1  # bridge the gap, stay in same sub-window
                else:
                    break  # gap too long or body changed — end sub-window
            else:
                break  # different body — end sub-window

        sub_windows.append((start + sub_start, start + j))
        i = j

    return sub_windows if sub_windows else [(start, end)]


def classify_mode(
    seg_right: np.ndarray,
    seg_left: np.ndarray,
    right_body: int | None,
    left_body: int | None,
) -> str:
    """Return interaction mode for a segment.

    seg_right / seg_left are bool[T] slices scoped to the segment.
    """
    both_hands_active = seg_right.any() and seg_left.any()
    if not both_hands_active:
        return "A"
    if right_body is None or left_body is None or right_body == left_body:
        return "B"
    return "C"


def _remap_paths(
    paths: list[str] | None, old_prefix: str, new_prefix: str
) -> list[str] | None:
    """Replace old_prefix with new_prefix in each path string."""
    if not paths:
        return paths
    return [p.replace(old_prefix, new_prefix, 1) if p else p for p in paths]


# Scalar per-body lists that are indexed [N_bodies] and must be filtered together
# with object_body_position / object_body_wxyz.
_PER_BODY_SCALAR_FIELDS = (
    "object_body_names",
    "safe_object_body_names",
    "object_mesh_paths",
    "object_urdf_paths",
    "object_mesh_radius",
)
# Timeseries with shape [T, N_bodies, *] that need body-axis filtering.
_PER_BODY_TS_FIELDS = ("object_body_position", "object_body_wxyz")
# Contact-part-id fields whose values are 1-indexed body IDs and must be remapped.
_CONTACT_PART_ID_FIELDS = (
    "mano_right_object_contact_part_ids",
    "mano_left_object_contact_part_ids",
)


def _object_prefix(body_name: str) -> str:
    """Strip trailing _body_N / _Body_N suffix to get the physical-object prefix."""
    return re.sub(r"_[Bb]ody_\d+$", "", body_name)


def _expand_to_object_siblings(body_1idx: set[int], body_names: list[str]) -> set[int]:
    """Expand a set of 1-indexed body IDs to include all sibling bodies of the
    same physical object (bodies sharing the same name prefix before _body_N).

    E.g. dominant body 1 ('dumbbell_body_0') also pulls in body 2 ('dumbbell_body_1').
    """
    prefix_to_ids: dict[str, list[int]] = {}
    for i, name in enumerate(body_names):
        prefix = _object_prefix(name)
        prefix_to_ids.setdefault(prefix, []).append(i + 1)  # 1-indexed

    expanded = set(body_1idx)
    for bid in list(body_1idx):
        if 1 <= bid <= len(body_names):
            prefix = _object_prefix(body_names[bid - 1])
            for sibling in prefix_to_ids.get(prefix, []):
                expanded.add(sibling)
    return expanded


def _filter_to_touched_bodies(
    d: dict, dominant_body_1idx: set[int] | None = None
) -> dict:
    """Keep only the object bodies for this segment; remap contact IDs.

    dominant_body_1idx: 1-indexed body IDs determined by majority-vote in
    segment_sequence (right_body / left_body).  Sibling bodies of the same
    physical object are automatically included.

    If dominant_body_1idx is None (no contact recorded), all bodies are kept.
    """
    body_names: list[str] = d.get("object_body_names") or []
    if not body_names:
        return d

    if dominant_body_1idx is None or not dominant_body_1idx:
        return d

    # Expand dominant IDs to include all sibling bodies of the same object.
    touched_1idx = _expand_to_object_siblings(dominant_body_1idx, body_names)

    # Sorted 0-indexed list of bodies to keep.
    n_bodies = len(body_names)
    touched_0idx: list[int] = sorted(v - 1 for v in touched_1idx if 1 <= v <= n_bodies)
    if not touched_0idx:
        return d

    # Build old-1idx → new-1idx remapping.
    old1_to_new1: dict[int, int] = {
        old0 + 1: new0 + 1 for new0, old0 in enumerate(touched_0idx)
    }

    # Filter scalar per-body lists.
    for field in _PER_BODY_SCALAR_FIELDS:
        val = d.get(field)
        if val is not None:
            d[field] = [val[i] for i in touched_0idx if i < len(val)]

    # Filter timeseries body axis: shape [T, N_bodies, *] → [T, len(touched), *].
    for field in _PER_BODY_TS_FIELDS:
        val = d.get(field)
        if val is None:
            continue
        d[field] = [
            [frame_bodies[i] for i in touched_0idx if i < len(frame_bodies)]
            for frame_bodies in val
        ]

    # Remap contact part IDs: 0 stays 0; unmapped old IDs (other objects) → 0.
    for field in _CONTACT_PART_ID_FIELDS:
        val = d.get(field)
        if val is None:
            continue
        d[field] = [[old1_to_new1.get(v, 0) for v in frame_ids] for frame_ids in val]

    return d


def slice_data(
    data: ManoSharpaData,
    start: int,
    end: int,
    parent_id: str,
    seg_idx: int,
    local_repo: Path | None = None,
    dominant_body_1idx: set[int] | None = None,
    pad_frames: int = 0,
) -> ManoSharpaData:
    """Return a new ManoSharpaData containing only frames [start:end].

    dominant_body_1idx: 1-indexed body IDs from majority-vote contact detection;
    only those objects (and their siblings) are kept in the output.

    If local_repo is provided, rewrites object_mesh_paths and object_urdf_paths
    so that OSMO-generated /workspace/video_to_data/ prefixes resolve locally.

    pad_frames: number of still frames to prepend (duplicating frame 0) and
    append (duplicating the last frame).  This gives support-surface reconstruction
    a guaranteed window of stationary object poses at both ends of each clip.
    """
    d = data.to_dict()
    for field_name in _TIMESERIES_FIELDS:
        if d.get(field_name) is not None:
            sliced = list(d[field_name][start:end])
            if pad_frames > 0 and sliced:
                sliced = [sliced[0]] * pad_frames + sliced + [sliced[-1]] * pad_frames
            d[field_name] = sliced
    d["sequence_id"] = f"{parent_id}_seg{seg_idx:03d}"

    _filter_to_touched_bodies(d, dominant_body_1idx=dominant_body_1idx)

    if local_repo is not None:
        old_prefix = "/workspace/video_to_data"
        new_prefix = str(local_repo)
        d["object_mesh_paths"] = _remap_paths(
            d.get("object_mesh_paths"), old_prefix, new_prefix
        )
        d["object_urdf_paths"] = _remap_paths(
            d.get("object_urdf_paths"), old_prefix, new_prefix
        )

    return ManoSharpaData(**d)


# ---------------------------------------------------------------------------
# Per-sequence segmentation
# ---------------------------------------------------------------------------


def segment_sequence(
    data: ManoSharpaData,
    threshold: float,
    gap_frames: int,
    min_frames: int,
) -> list[dict]:
    """Compute segment metadata dicts for one sequence.

    Returns a list of dicts with keys matching MANIFEST_COLUMNS
    (excluding segment_id, which is filled by the caller).
    """
    fps = data.fps or 30.0
    tips_r = np.array(data.mano_right_tips_distance)  # [T, 5]
    tips_l = np.array(data.mano_left_tips_distance)  # [T, 5]
    cpid_r = np.array(data.mano_right_object_contact_part_ids)  # [T, 16]
    cpid_l = np.array(data.mano_left_object_contact_part_ids)  # [T, 16]

    active_r = detect_contact(tips_r, threshold)
    active_l = detect_contact(tips_l, threshold)
    active_r = fill_gaps(active_r, gap_frames)
    active_l = fill_gaps(active_l, gap_frames)

    windows = find_active_windows(active_r, active_l)

    # Split windows that span multiple objects (fast object transitions get
    # bridged by gap-filling above; split_by_object undoes that).
    per_frame_body = _per_frame_dominant_body(cpid_r, cpid_l)
    refined: list[tuple[int, int]] = []
    for start, end in windows:
        refined.extend(split_by_object(start, end, per_frame_body, gap_frames))

    segments: list[dict] = []
    for start, end in refined:
        num_frames = end - start
        if num_frames < min_frames:
            continue

        seg_r = active_r[start:end]
        seg_l = active_l[start:end]
        right_body = get_dominant_body(cpid_r[start:end], seg_r)
        left_body = get_dominant_body(cpid_l[start:end], seg_l)
        mode = classify_mode(seg_r, seg_l, right_body, left_body)
        dominant = {b for b in (right_body, left_body) if b is not None}

        segments.append(
            {
                "parent_sequence_id": data.sequence_id,
                "segment_idx": len(segments),
                "mode": mode,
                "start_frame": start,
                "end_frame": end,
                "num_frames": num_frames,
                "duration_s": round(num_frames / fps, 2),
                "object_name": data.object_name or "",
                # Internal: passed to slice_data but excluded from manifest CSV.
                "_dominant_bodies": dominant,
            }
        )
    return segments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment retargeted sequences into atomic hand-object interaction clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to a <dataset>_processed/ parquet directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to {input_dir.name}_segmented/ next to input_dir.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Fingertip-to-surface distance threshold in metres for contact detection.",
    )
    parser.add_argument(
        "--gap_frames",
        type=int,
        default=15,
        help="Fill contact gaps shorter than this many frames (0.5 s at 30 Hz).",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=30,
        help="Drop segments shorter than this many frames (1 s at 30 Hz).",
    )
    parser.add_argument(
        "--pad_frames",
        type=int,
        default=SEGMENT_STILL_PADDING_FRAMES,
        help=(
            "Still frames to prepend/append to each segment (duplicate of first/last frame). "
            "Ensures support-surface reconstruction has a stationary object window at both ends. "
            f"Default: {SEGMENT_STILL_PADDING_FRAMES}."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print segment manifest only; do not write parquet files.",
    )
    parser.add_argument(
        "--local_repo",
        type=Path,
        default=None,
        help=(
            "When set, rewrites /workspace/video_to_data/ in object_mesh_paths and "
            "object_urdf_paths to this local monorepo root. Leave unset (default) when "
            "vis_retargeted.py will be run inside the Docker container where "
            "/workspace/video_to_data/ is already correctly mounted."
        ),
    )
    add_sequence_filter_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir: Path = args.output_dir or (
        args.input_dir.parent / f"{args.input_dir.name}_segmented"
    )

    sequence_ids = list_sequence_ids(str(args.input_dir))
    sequence_ids = filter_sequence_ids(sequence_ids, args)
    if not sequence_ids:
        print("No sequences matched the filter.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(sequence_ids)} sequence(s).")
    print(f"  input:     {args.input_dir}")
    if not args.dry_run:
        print(f"  output:    {output_dir}")
    print(
        f"  threshold: {args.threshold} m  |  gap_frames: {args.gap_frames}"
        f"  |  min_frames: {args.min_frames}"
    )

    manifest_path = output_dir.parent / "segment_manifest.csv"
    manifest_writer = None
    manifest_file = None

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = open(manifest_path, "w", newline="")
        manifest_writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        manifest_writer.writeheader()

    total_segs = 0

    for seq_id in tqdm(sequence_ids, desc="sequences"):
        data = ManoSharpaData.from_parquet(
            root_path=str(args.input_dir),
            filters=[("sequence_id", "=", seq_id)],
        )

        segments = segment_sequence(
            data, args.threshold, args.gap_frames, args.min_frames
        )

        for seg in segments:
            seg_id = f"{seg['parent_sequence_id']}_seg{seg['segment_idx']:03d}"
            # Exclude internal keys (prefixed with _) from the manifest row.
            row = {
                "segment_id": seg_id,
                **{k: v for k, v in seg.items() if not k.startswith("_")},
            }

            if args.dry_run:
                print(
                    f"  {seg_id}  mode={seg['mode']}"
                    f"  [{seg['start_frame']}:{seg['end_frame']}]"
                    f"  {seg['duration_s']:.1f}s"
                )
            else:
                sliced = slice_data(
                    data,
                    seg["start_frame"],
                    seg["end_frame"],
                    seg["parent_sequence_id"],
                    seg["segment_idx"],
                    local_repo=args.local_repo,
                    dominant_body_1idx=seg.get("_dominant_bodies"),
                    pad_frames=args.pad_frames,
                )
                sliced.save_to_parquet(
                    root_path=str(output_dir),
                    partition_cols=DEFAULT_PARTITION_COLS,
                )
                manifest_writer.writerow(row)

        total_segs += len(segments)

    if manifest_file is not None:
        manifest_file.close()

    print(f"\nTotal segments: {total_segs} from {len(sequence_ids)} sequence(s).")
    if not args.dry_run:
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
