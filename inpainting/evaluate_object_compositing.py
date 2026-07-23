"""Evaluate an estimated object-occlusion condition against a GT reference.

The robot render is fixed for both conditions.  This evaluator reconstructs
the GT and candidate visible-robot decisions from their respective occluder
mask/depth pairs using the production 3 mm guard, then compares those binary
decisions.  Input ``.npy`` arrays are memory-mapped and processed one frame at
a time so full-resolution videos do not need to be materialized in RAM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable
import uuid

import numpy as np

from .contracts import ContractError


OBJECT_COMPOSITING_EVALUATION_SCHEMA = "v2d.inpainting.object-compositing-evaluation/v1"
DEPTH_GUARD_M = 0.003


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _iou(intersection: int, union: int) -> float:
    # Identical empty masks are a perfect set match.  Recording the convention
    # in the output keeps empty frames from silently becoming NaN in JSON.
    return 1.0 if union == 0 else float(intersection / union)


def _distribution(values: list[int | float | None]) -> dict[str, int | float | None]:
    finite = np.asarray(
        [value for value in values if value is not None], dtype=np.float64
    )
    if not finite.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    if not np.isfinite(finite).all():
        raise RuntimeError("Internal metric summary received a non-finite value")
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95, method="linear")),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _error_metric(
    *,
    count: int,
    decision_denominator_name: str,
    decision_denominator: int,
    total_robot_pixels: int,
) -> dict[str, Any]:
    return {
        "count": int(count),
        "decision_denominator": {
            "name": decision_denominator_name,
            "pixel_count": int(decision_denominator),
        },
        "rate_decision_denominator": _ratio(count, decision_denominator),
        "total_robot_pixel_denominator": int(total_robot_pixels),
        "rate_total_robot_pixels": _ratio(count, total_robot_pixels),
    }


def _visibility_record(
    *,
    robot_pixels: int,
    gt_visible: int,
    candidate_visible: int,
    both_visible: int,
    both_occluded: int,
    false_visible: int,
    false_occluded: int,
) -> dict[str, Any]:
    gt_occluded = robot_pixels - gt_visible
    candidate_occluded = robot_pixels - candidate_visible
    visible_union = both_visible + false_visible + false_occluded
    disagreement = false_visible + false_occluded
    return {
        "robot_pixel_count": int(robot_pixels),
        "gt_visible_pixel_count": int(gt_visible),
        "gt_occluded_pixel_count": int(gt_occluded),
        "candidate_visible_pixel_count": int(candidate_visible),
        "candidate_occluded_pixel_count": int(candidate_occluded),
        "both_visible_pixel_count": int(both_visible),
        "both_occluded_pixel_count": int(both_occluded),
        "visible_union_pixel_count": int(visible_union),
        "visible_iou": _iou(both_visible, visible_union),
        "false_visible": _error_metric(
            count=false_visible,
            decision_denominator_name="gt_occluded_robot_pixels",
            decision_denominator=gt_occluded,
            total_robot_pixels=robot_pixels,
        ),
        "false_occluded": _error_metric(
            count=false_occluded,
            decision_denominator_name="gt_visible_robot_pixels",
            decision_denominator=gt_visible,
            total_robot_pixels=robot_pixels,
        ),
        "decision_disagreement": {
            "count": int(disagreement),
            "rate_total_robot_pixels": _ratio(disagreement, robot_pixels),
        },
    }


def _object_mask_record(
    *,
    gt_pixels: int,
    candidate_pixels: int,
    intersection: int,
    union: int,
) -> dict[str, int | float]:
    return {
        "gt_pixel_count": int(gt_pixels),
        "candidate_pixel_count": int(candidate_pixels),
        "intersection_pixel_count": int(intersection),
        "union_pixel_count": int(union),
        "iou": _iou(intersection, union),
    }


def _depth_record(
    *, overlap_pixels: int, absolute_error_sum_m: float
) -> dict[str, Any]:
    return {
        "overlap_pixel_count": int(overlap_pixels),
        "absolute_error_sum_m": float(absolute_error_sum_m),
        "mae_m": (
            float(absolute_error_sum_m / overlap_pixels) if overlap_pixels else None
        ),
    }


def _validate_basic_array_contracts(
    *,
    robot_mask: np.ndarray,
    robot_depth: np.ndarray,
    gt_occluder_mask: np.ndarray,
    gt_occluder_depth: np.ndarray,
    candidate_occluder_mask: np.ndarray,
    candidate_occluder_depth: np.ndarray,
) -> tuple[int, int, int]:
    arrays = {
        "robot_mask": robot_mask,
        "robot_depth": robot_depth,
        "gt_occluder_mask": gt_occluder_mask,
        "gt_occluder_depth": gt_occluder_depth,
        "candidate_occluder_mask": candidate_occluder_mask,
        "candidate_occluder_depth": candidate_occluder_depth,
    }
    shape = robot_mask.shape
    if len(shape) != 3 or any(dimension <= 0 for dimension in shape):
        raise ContractError(
            "robot_mask must have a non-empty (frame_count, height, width) shape"
        )
    for name, array in arrays.items():
        if array.shape != shape:
            raise ContractError(f"{name} must have shape {shape}, got {array.shape}")
    for name in ("robot_mask", "gt_occluder_mask", "candidate_occluder_mask"):
        if arrays[name].dtype != np.dtype(np.bool_):
            raise ContractError(
                f"{name} must have boolean dtype, got {arrays[name].dtype}"
            )
    for name in ("robot_depth", "gt_occluder_depth", "candidate_occluder_depth"):
        if arrays[name].dtype != np.dtype(np.float32):
            raise ContractError(
                f"{name} must have float32 dtype, got {arrays[name].dtype}"
            )
    return int(shape[0]), int(shape[1]), int(shape[2])


def _validate_depth_frame(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    label: str,
    frame_index: int,
) -> None:
    masked_depth = depth[mask]
    if masked_depth.size and (
        not np.isfinite(masked_depth).all() or np.any(masked_depth <= 0.0)
    ):
        raise ContractError(
            f"{label} frame {frame_index} has non-finite or non-positive masked values"
        )
    if not np.isposinf(depth[~mask]).all():
        raise ContractError(
            f"{label} frame {frame_index} must be +inf outside its mask"
        )


def _visible_robot_mask(
    robot_mask: np.ndarray,
    robot_depth: np.ndarray,
    occluder_mask: np.ndarray,
    occluder_depth: np.ndarray,
) -> np.ndarray:
    """Reproduce the production object-depth compositing decision."""

    return robot_mask & (
        ~occluder_mask | (robot_depth <= occluder_depth + DEPTH_GUARD_M)
    )


def _rank_frames(
    records: list[dict[str, Any]],
    *,
    value: Callable[[dict[str, Any]], int | float | None],
    limit: int,
    higher_is_worse: bool,
) -> list[dict[str, int | float]]:
    ranked: list[tuple[int, int | float]] = []
    for record in records:
        metric = value(record)
        if metric is not None:
            ranked.append((int(record["frame_index"]), metric))
    if higher_is_worse:
        ranked.sort(key=lambda item: (-float(item[1]), item[0]))
    else:
        ranked.sort(key=lambda item: (float(item[1]), item[0]))
    return [
        {"frame_index": frame_index, "value": metric}
        for frame_index, metric in ranked[:limit]
    ]


def _rank_frame_pairs(
    records: list[dict[str, Any]], *, limit: int
) -> list[dict[str, int | float]]:
    eligible = [
        record
        for record in records
        if record["transition_disagreement_rate"] is not None
    ]
    eligible.sort(
        key=lambda record: (
            -float(record["transition_disagreement_rate"]),
            -int(record["transition_disagreement_count"]),
            int(record["from_frame"]),
        )
    )
    return [
        {
            "from_frame": int(record["from_frame"]),
            "to_frame": int(record["to_frame"]),
            "transition_disagreement_count": int(
                record["transition_disagreement_count"]
            ),
            "transition_disagreement_rate": float(
                record["transition_disagreement_rate"]
            ),
        }
        for record in eligible[:limit]
    ]


def _self_checks(
    *,
    visibility: dict[str, int],
    object_mask: dict[str, int],
    temporal: dict[str, int],
    frame_records: list[dict[str, Any]],
    pair_records: list[dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, dict[str, int | bool]] = {}

    def check(name: str, actual: int, expected: int) -> None:
        checks[name] = {
            "passed": actual == expected,
            "actual": int(actual),
            "expected": int(expected),
        }

    check(
        "visibility_confusion_partitions_robot_pixels",
        visibility["both_visible"]
        + visibility["both_occluded"]
        + visibility["false_visible"]
        + visibility["false_occluded"],
        visibility["robot"],
    )
    check(
        "gt_visible_partition",
        visibility["both_visible"] + visibility["false_occluded"],
        visibility["gt_visible"],
    )
    check(
        "candidate_visible_partition",
        visibility["both_visible"] + visibility["false_visible"],
        visibility["candidate_visible"],
    )
    check(
        "visible_union_partition",
        visibility["both_visible"]
        + visibility["false_visible"]
        + visibility["false_occluded"],
        visibility["visible_union"],
    )
    check(
        "visibility_disagreement_partition",
        visibility["false_visible"] + visibility["false_occluded"],
        visibility["disagreement"],
    )
    check(
        "visible_masks_are_subsets_of_robot_mask",
        visibility["visible_outside_robot"],
        0,
    )
    check(
        "object_union_partition",
        object_mask["intersection"]
        + object_mask["gt_only"]
        + object_mask["candidate_only"],
        object_mask["union"],
    )
    check(
        "depth_overlap_matches_object_mask_intersection",
        object_mask["depth_overlap"],
        object_mask["intersection"],
    )
    check(
        "temporal_partition_of_persistent_pixels",
        temporal["both_transition"]
        + temporal["neither_transition"]
        + temporal["spurious"]
        + temporal["missed"],
        temporal["persistent"],
    )
    check(
        "temporal_disagreement_partition",
        temporal["spurious"] + temporal["missed"],
        temporal["disagreement"],
    )
    check(
        "gt_transition_partition",
        temporal["both_transition"] + temporal["missed"],
        temporal["gt_transition"],
    )
    check(
        "candidate_transition_partition",
        temporal["both_transition"] + temporal["spurious"],
        temporal["candidate_transition"],
    )
    check(
        "per_frame_robot_pixels_sum_to_micro",
        sum(record["visibility"]["robot_pixel_count"] for record in frame_records),
        visibility["robot"],
    )
    check(
        "per_frame_false_visible_sum_to_micro",
        sum(record["visibility"]["false_visible"]["count"] for record in frame_records),
        visibility["false_visible"],
    )
    check(
        "per_frame_false_occluded_sum_to_micro",
        sum(
            record["visibility"]["false_occluded"]["count"] for record in frame_records
        ),
        visibility["false_occluded"],
    )
    check(
        "per_pair_persistent_pixels_sum_to_micro",
        sum(record["persistent_pixel_count"] for record in pair_records),
        temporal["persistent"],
    )
    check(
        "per_pair_transition_disagreement_sum_to_micro",
        sum(record["transition_disagreement_count"] for record in pair_records),
        temporal["disagreement"],
    )

    failed = [name for name, result in checks.items() if not result["passed"]]
    if failed:
        raise RuntimeError("Internal evaluator invariant failed: " + ", ".join(failed))
    return {
        "all_passed": True,
        "checks": checks,
    }


def evaluate_arrays(
    *,
    robot_mask: np.ndarray,
    robot_depth: np.ndarray,
    gt_occluder_mask: np.ndarray,
    gt_occluder_depth: np.ndarray,
    candidate_occluder_mask: np.ndarray,
    candidate_occluder_depth: np.ndarray,
    sequence_id: str,
    candidate_name: str,
    worst_frame_count: int = 10,
) -> dict[str, Any]:
    """Evaluate already-open arrays with bounded, frame-wise working memory."""

    if not sequence_id.strip():
        raise ValueError("sequence_id must not be empty")
    if not candidate_name.strip():
        raise ValueError("candidate_name must not be empty")
    if (
        isinstance(worst_frame_count, bool)
        or not isinstance(worst_frame_count, int)
        or worst_frame_count <= 0
    ):
        raise ValueError("worst_frame_count must be a positive integer")

    frame_count, height, width = _validate_basic_array_contracts(
        robot_mask=robot_mask,
        robot_depth=robot_depth,
        gt_occluder_mask=gt_occluder_mask,
        gt_occluder_depth=gt_occluder_depth,
        candidate_occluder_mask=candidate_occluder_mask,
        candidate_occluder_depth=candidate_occluder_depth,
    )

    visibility_counts = {
        "robot": 0,
        "gt_visible": 0,
        "candidate_visible": 0,
        "both_visible": 0,
        "both_occluded": 0,
        "false_visible": 0,
        "false_occluded": 0,
        "visible_union": 0,
        "disagreement": 0,
        "visible_outside_robot": 0,
    }
    object_counts = {
        "gt": 0,
        "candidate": 0,
        "intersection": 0,
        "union": 0,
        "gt_only": 0,
        "candidate_only": 0,
        "depth_overlap": 0,
    }
    temporal_counts = {
        "persistent": 0,
        "gt_transition": 0,
        "candidate_transition": 0,
        "both_transition": 0,
        "neither_transition": 0,
        "spurious": 0,
        "missed": 0,
        "disagreement": 0,
    }
    depth_absolute_error_sum_m = 0.0
    frame_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []

    previous_robot: np.ndarray | None = None
    previous_gt_visible: np.ndarray | None = None
    previous_candidate_visible: np.ndarray | None = None

    for frame_index in range(frame_count):
        frame_robot_mask = robot_mask[frame_index]
        frame_robot_depth = robot_depth[frame_index]
        frame_gt_mask = gt_occluder_mask[frame_index]
        frame_gt_depth = gt_occluder_depth[frame_index]
        frame_candidate_mask = candidate_occluder_mask[frame_index]
        frame_candidate_depth = candidate_occluder_depth[frame_index]

        _validate_depth_frame(
            frame_robot_depth,
            frame_robot_mask,
            label="robot_depth",
            frame_index=frame_index,
        )
        _validate_depth_frame(
            frame_gt_depth,
            frame_gt_mask,
            label="gt_occluder_depth",
            frame_index=frame_index,
        )
        _validate_depth_frame(
            frame_candidate_depth,
            frame_candidate_mask,
            label="candidate_occluder_depth",
            frame_index=frame_index,
        )

        gt_visible_mask = _visible_robot_mask(
            frame_robot_mask,
            frame_robot_depth,
            frame_gt_mask,
            frame_gt_depth,
        )
        candidate_visible_mask = _visible_robot_mask(
            frame_robot_mask,
            frame_robot_depth,
            frame_candidate_mask,
            frame_candidate_depth,
        )

        both_visible_mask = gt_visible_mask & candidate_visible_mask
        false_visible_mask = candidate_visible_mask & ~gt_visible_mask
        false_occluded_mask = gt_visible_mask & ~candidate_visible_mask
        both_occluded_mask = (
            frame_robot_mask & ~gt_visible_mask & ~candidate_visible_mask
        )
        visible_union_mask = gt_visible_mask | candidate_visible_mask

        frame_visibility = {
            "robot": int(np.count_nonzero(frame_robot_mask)),
            "gt_visible": int(np.count_nonzero(gt_visible_mask)),
            "candidate_visible": int(np.count_nonzero(candidate_visible_mask)),
            "both_visible": int(np.count_nonzero(both_visible_mask)),
            "both_occluded": int(np.count_nonzero(both_occluded_mask)),
            "false_visible": int(np.count_nonzero(false_visible_mask)),
            "false_occluded": int(np.count_nonzero(false_occluded_mask)),
            "visible_union": int(np.count_nonzero(visible_union_mask)),
        }
        frame_visibility["disagreement"] = (
            frame_visibility["false_visible"] + frame_visibility["false_occluded"]
        )
        for key in frame_visibility:
            visibility_counts[key] += frame_visibility[key]
        visibility_counts["visible_outside_robot"] += int(
            np.count_nonzero(
                (gt_visible_mask | candidate_visible_mask) & ~frame_robot_mask
            )
        )

        object_intersection_mask = frame_gt_mask & frame_candidate_mask
        object_union_mask = frame_gt_mask | frame_candidate_mask
        frame_object = {
            "gt": int(np.count_nonzero(frame_gt_mask)),
            "candidate": int(np.count_nonzero(frame_candidate_mask)),
            "intersection": int(np.count_nonzero(object_intersection_mask)),
            "union": int(np.count_nonzero(object_union_mask)),
            "gt_only": int(np.count_nonzero(frame_gt_mask & ~frame_candidate_mask)),
            "candidate_only": int(
                np.count_nonzero(frame_candidate_mask & ~frame_gt_mask)
            ),
        }
        for key in (
            "gt",
            "candidate",
            "intersection",
            "union",
            "gt_only",
            "candidate_only",
        ):
            object_counts[key] += frame_object[key]

        overlap_pixels = frame_object["intersection"]
        if overlap_pixels:
            absolute_depth_error = np.abs(
                np.asarray(
                    frame_candidate_depth[object_intersection_mask], dtype=np.float64
                )
                - np.asarray(frame_gt_depth[object_intersection_mask], dtype=np.float64)
            )
            frame_depth_absolute_error_sum_m = float(
                np.sum(absolute_depth_error, dtype=np.float64)
            )
        else:
            frame_depth_absolute_error_sum_m = 0.0
        object_counts["depth_overlap"] += overlap_pixels
        depth_absolute_error_sum_m += frame_depth_absolute_error_sum_m

        visibility_record = _visibility_record(
            robot_pixels=frame_visibility["robot"],
            gt_visible=frame_visibility["gt_visible"],
            candidate_visible=frame_visibility["candidate_visible"],
            both_visible=frame_visibility["both_visible"],
            both_occluded=frame_visibility["both_occluded"],
            false_visible=frame_visibility["false_visible"],
            false_occluded=frame_visibility["false_occluded"],
        )
        frame_records.append(
            {
                "frame_index": frame_index,
                "visibility": visibility_record,
                "occluder_mask": _object_mask_record(
                    gt_pixels=frame_object["gt"],
                    candidate_pixels=frame_object["candidate"],
                    intersection=frame_object["intersection"],
                    union=frame_object["union"],
                ),
                "overlap_depth": _depth_record(
                    overlap_pixels=overlap_pixels,
                    absolute_error_sum_m=frame_depth_absolute_error_sum_m,
                ),
            }
        )

        if previous_robot is not None:
            assert previous_gt_visible is not None
            assert previous_candidate_visible is not None
            persistent_mask = previous_robot & frame_robot_mask
            gt_transition_mask = persistent_mask & (
                previous_gt_visible != gt_visible_mask
            )
            candidate_transition_mask = persistent_mask & (
                previous_candidate_visible != candidate_visible_mask
            )
            both_transition_mask = gt_transition_mask & candidate_transition_mask
            spurious_transition_mask = candidate_transition_mask & ~gt_transition_mask
            missed_transition_mask = gt_transition_mask & ~candidate_transition_mask
            neither_transition_mask = (
                persistent_mask & ~gt_transition_mask & ~candidate_transition_mask
            )
            pair = {
                "from_frame": frame_index - 1,
                "to_frame": frame_index,
                "persistent_pixel_count": int(np.count_nonzero(persistent_mask)),
                "gt_transition_count": int(np.count_nonzero(gt_transition_mask)),
                "candidate_transition_count": int(
                    np.count_nonzero(candidate_transition_mask)
                ),
                "both_transition_count": int(np.count_nonzero(both_transition_mask)),
                "neither_transition_count": int(
                    np.count_nonzero(neither_transition_mask)
                ),
                "spurious_candidate_transition_count": int(
                    np.count_nonzero(spurious_transition_mask)
                ),
                "missed_candidate_transition_count": int(
                    np.count_nonzero(missed_transition_mask)
                ),
            }
            pair["transition_disagreement_count"] = (
                pair["spurious_candidate_transition_count"]
                + pair["missed_candidate_transition_count"]
            )
            pair["transition_disagreement_rate"] = _ratio(
                pair["transition_disagreement_count"],
                pair["persistent_pixel_count"],
            )
            pair_records.append(pair)
            temporal_counts["persistent"] += pair["persistent_pixel_count"]
            temporal_counts["gt_transition"] += pair["gt_transition_count"]
            temporal_counts["candidate_transition"] += pair[
                "candidate_transition_count"
            ]
            temporal_counts["both_transition"] += pair["both_transition_count"]
            temporal_counts["neither_transition"] += pair["neither_transition_count"]
            temporal_counts["spurious"] += pair["spurious_candidate_transition_count"]
            temporal_counts["missed"] += pair["missed_candidate_transition_count"]
            temporal_counts["disagreement"] += pair["transition_disagreement_count"]

        previous_robot = frame_robot_mask
        previous_gt_visible = gt_visible_mask
        previous_candidate_visible = candidate_visible_mask

    micro_visibility = _visibility_record(
        robot_pixels=visibility_counts["robot"],
        gt_visible=visibility_counts["gt_visible"],
        candidate_visible=visibility_counts["candidate_visible"],
        both_visible=visibility_counts["both_visible"],
        both_occluded=visibility_counts["both_occluded"],
        false_visible=visibility_counts["false_visible"],
        false_occluded=visibility_counts["false_occluded"],
    )
    micro_object_mask = _object_mask_record(
        gt_pixels=object_counts["gt"],
        candidate_pixels=object_counts["candidate"],
        intersection=object_counts["intersection"],
        union=object_counts["union"],
    )
    micro_depth = _depth_record(
        overlap_pixels=object_counts["depth_overlap"],
        absolute_error_sum_m=depth_absolute_error_sum_m,
    )

    per_frame_summary = {
        "visible_iou": _distribution(
            [record["visibility"]["visible_iou"] for record in frame_records]
        ),
        "decision_disagreement_count": _distribution(
            [
                record["visibility"]["decision_disagreement"]["count"]
                for record in frame_records
            ]
        ),
        "decision_disagreement_rate_total_robot_pixels": _distribution(
            [
                record["visibility"]["decision_disagreement"]["rate_total_robot_pixels"]
                for record in frame_records
            ]
        ),
        "false_visible_count": _distribution(
            [record["visibility"]["false_visible"]["count"] for record in frame_records]
        ),
        "false_visible_rate_decision_denominator": _distribution(
            [
                record["visibility"]["false_visible"]["rate_decision_denominator"]
                for record in frame_records
            ]
        ),
        "false_visible_rate_total_robot_pixels": _distribution(
            [
                record["visibility"]["false_visible"]["rate_total_robot_pixels"]
                for record in frame_records
            ]
        ),
        "false_occluded_count": _distribution(
            [
                record["visibility"]["false_occluded"]["count"]
                for record in frame_records
            ]
        ),
        "false_occluded_rate_decision_denominator": _distribution(
            [
                record["visibility"]["false_occluded"]["rate_decision_denominator"]
                for record in frame_records
            ]
        ),
        "false_occluded_rate_total_robot_pixels": _distribution(
            [
                record["visibility"]["false_occluded"]["rate_total_robot_pixels"]
                for record in frame_records
            ]
        ),
        "occluder_mask_iou": _distribution(
            [record["occluder_mask"]["iou"] for record in frame_records]
        ),
        "overlap_depth_mae_m": _distribution(
            [record["overlap_depth"]["mae_m"] for record in frame_records]
        ),
    }

    worst_frames = {
        "highest_decision_disagreement_count": _rank_frames(
            frame_records,
            value=lambda record: record["visibility"]["decision_disagreement"]["count"],
            limit=worst_frame_count,
            higher_is_worse=True,
        ),
        "highest_decision_disagreement_rate_total_robot_pixels": _rank_frames(
            frame_records,
            value=lambda record: record["visibility"]["decision_disagreement"][
                "rate_total_robot_pixels"
            ],
            limit=worst_frame_count,
            higher_is_worse=True,
        ),
        "highest_false_visible_count": _rank_frames(
            frame_records,
            value=lambda record: record["visibility"]["false_visible"]["count"],
            limit=worst_frame_count,
            higher_is_worse=True,
        ),
        "highest_false_visible_rate_decision_denominator": _rank_frames(
            frame_records,
            value=lambda record: record["visibility"]["false_visible"][
                "rate_decision_denominator"
            ],
            limit=worst_frame_count,
            higher_is_worse=True,
        ),
        "highest_false_occluded_count": _rank_frames(
            frame_records,
            value=lambda record: record["visibility"]["false_occluded"]["count"],
            limit=worst_frame_count,
            higher_is_worse=True,
        ),
        "highest_false_occluded_rate_decision_denominator": _rank_frames(
            frame_records,
            value=lambda record: record["visibility"]["false_occluded"][
                "rate_decision_denominator"
            ],
            limit=worst_frame_count,
            higher_is_worse=True,
        ),
        "lowest_visible_iou": _rank_frames(
            frame_records,
            value=lambda record: record["visibility"]["visible_iou"],
            limit=worst_frame_count,
            higher_is_worse=False,
        ),
        "lowest_occluder_mask_iou": _rank_frames(
            frame_records,
            value=lambda record: record["occluder_mask"]["iou"],
            limit=worst_frame_count,
            higher_is_worse=False,
        ),
        "highest_overlap_depth_mae_m": _rank_frames(
            frame_records,
            value=lambda record: record["overlap_depth"]["mae_m"],
            limit=worst_frame_count,
            higher_is_worse=True,
        ),
    }

    self_checks = _self_checks(
        visibility=visibility_counts,
        object_mask=object_counts,
        temporal=temporal_counts,
        frame_records=frame_records,
        pair_records=pair_records,
    )

    return {
        "schema_version": OBJECT_COMPOSITING_EVALUATION_SCHEMA,
        "state": "complete",
        "sequence_id": sequence_id,
        "candidate_name": candidate_name,
        "geometry": {
            "frame_count": frame_count,
            "height": height,
            "width": width,
        },
        "definitions": {
            "camera_depth": "positive metric camera-z in metres; invalid is +inf",
            "depth_guard_m": DEPTH_GUARD_M,
            "visible_robot": (
                "robot_mask & (~occluder_mask | "
                "robot_depth <= occluder_depth + depth_guard_m)"
            ),
            "false_visible": "candidate visible and GT occluded at a robot pixel",
            "false_occluded": "candidate occluded and GT visible at a robot pixel",
            "decision_denominator_rates": (
                "false-visible uses GT-occluded robot pixels; false-occluded uses "
                "GT-visible robot pixels"
            ),
            "persistent_pixel": (
                "the same image pixel is inside the robot mask in both adjacent frames"
            ),
            "temporal_transition": (
                "visible/occluded state changes across adjacent frames at a "
                "persistent pixel"
            ),
            "overlap_depth_mae": (
                "mean absolute metric depth error where both GT and candidate "
                "occluder masks are true"
            ),
            "empty_mask_iou": 1.0,
        },
        "metrics": {
            "visibility": {
                "micro": micro_visibility,
                "per_frame_summary": per_frame_summary,
            },
            "persistent_pixel_temporal_transition": {
                "adjacent_frame_pair_count": len(pair_records),
                "persistent_pixel_pair_count": temporal_counts["persistent"],
                "gt_transition_count": temporal_counts["gt_transition"],
                "candidate_transition_count": temporal_counts["candidate_transition"],
                "both_transition_count": temporal_counts["both_transition"],
                "spurious_candidate_transition_count": temporal_counts["spurious"],
                "missed_candidate_transition_count": temporal_counts["missed"],
                "transition_disagreement_count": temporal_counts["disagreement"],
                "transition_disagreement_rate": _ratio(
                    temporal_counts["disagreement"], temporal_counts["persistent"]
                ),
                "per_frame_pair_summary": {
                    "transition_disagreement_count": _distribution(
                        [
                            record["transition_disagreement_count"]
                            for record in pair_records
                        ]
                    ),
                    "transition_disagreement_rate": _distribution(
                        [
                            record["transition_disagreement_rate"]
                            for record in pair_records
                        ]
                    ),
                },
                "worst_frame_pairs": _rank_frame_pairs(
                    pair_records, limit=worst_frame_count
                ),
                "per_frame_pair": pair_records,
            },
            "occluder_mask": {
                "micro": micro_object_mask,
            },
            "overlap_depth": {
                "micro": micro_depth,
            },
        },
        "per_frame": frame_records,
        "worst_frames": worst_frames,
        "self_checks": self_checks,
    }


def _open_npy_memmap(path: Path, *, label: str) -> np.ndarray:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ContractError(f"Could not memory-map {label}: {path}: {exc}") from exc
    if not isinstance(array, np.ndarray):
        close = getattr(array, "close", None)
        if close is not None:
            close()
        raise ContractError(f"{label} must be a single uncompressed .npy array: {path}")
    return array


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(
    path: Path, expected_signature: tuple[int, int, int, int]
) -> dict[str, Any]:
    if _stat_signature(path) != expected_signature:
        raise ContractError(f"Evaluation input changed while being read: {path}")
    digest = _sha256(path)
    if _stat_signature(path) != expected_signature:
        raise ContractError(f"Evaluation input changed while being hashed: {path}")
    return {
        "path": str(path),
        "size_bytes": expected_signature[2],
        "sha256": digest,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        encoded = (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # A same-directory hard link publishes the fully flushed inode and
            # atomically fails if another writer created the destination after
            # our initial existence check.  Plain os.replace would silently
            # clobber that concurrently created file.
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to replace existing output: {path}"
                ) from exc
            temporary.unlink()
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            # The rename itself remains atomic on supported local filesystems;
            # some mounted filesystems do not permit directory fsync.
            pass
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def evaluate_files(
    *,
    robot_mask_path: Path,
    robot_depth_path: Path,
    gt_occluder_mask_path: Path,
    gt_occluder_depth_path: Path,
    candidate_occluder_mask_path: Path,
    candidate_occluder_depth_path: Path,
    output_path: Path,
    sequence_id: str,
    candidate_name: str,
    worst_frame_count: int = 10,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Memory-map, evaluate, fingerprint, and atomically publish one report."""

    inputs = {
        "robot_mask": Path(robot_mask_path).expanduser().resolve(),
        "robot_depth": Path(robot_depth_path).expanduser().resolve(),
        "gt_occluder_mask": Path(gt_occluder_mask_path).expanduser().resolve(),
        "gt_occluder_depth": Path(gt_occluder_depth_path).expanduser().resolve(),
        "candidate_occluder_mask": Path(candidate_occluder_mask_path)
        .expanduser()
        .resolve(),
        "candidate_occluder_depth": Path(candidate_occluder_depth_path)
        .expanduser()
        .resolve(),
    }
    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix.lower() != ".json":
        raise ValueError("Evaluation output must use a .json suffix")
    if output_path in inputs.values():
        raise ValueError("Evaluation output must not alias an input")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing output: {output_path}")
    for label, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} input: {path}")

    signatures = {label: _stat_signature(path) for label, path in inputs.items()}
    arrays = {
        label: _open_npy_memmap(path, label=label) for label, path in inputs.items()
    }
    payload = evaluate_arrays(
        robot_mask=arrays["robot_mask"],
        robot_depth=arrays["robot_depth"],
        gt_occluder_mask=arrays["gt_occluder_mask"],
        gt_occluder_depth=arrays["gt_occluder_depth"],
        candidate_occluder_mask=arrays["candidate_occluder_mask"],
        candidate_occluder_depth=arrays["candidate_occluder_depth"],
        sequence_id=sequence_id,
        candidate_name=candidate_name,
        worst_frame_count=worst_frame_count,
    )

    # Hash each unique file once; comparing a candidate directly with the GT
    # bundle is a useful zero-error self-test and may intentionally reuse paths.
    artifact_cache: dict[Path, dict[str, Any]] = {}
    for label, path in inputs.items():
        if _stat_signature(path) != signatures[label]:
            raise ContractError(f"Evaluation input changed while being read: {path}")
        if path not in artifact_cache:
            artifact_cache[path] = _artifact(path, signatures[label])
    payload["inputs"] = {label: artifact_cache[path] for label, path in inputs.items()}
    for label, path in inputs.items():
        if _stat_signature(path) != signatures[label]:
            raise ContractError(f"Evaluation input changed while being hashed: {path}")
    _write_json_atomic(output_path, payload, overwrite=overwrite)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-mask", required=True, type=Path)
    parser.add_argument("--robot-depth", required=True, type=Path)
    parser.add_argument(
        "--gt-occluder-mask",
        "--ground-truth-occluder-mask",
        dest="gt_occluder_mask",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--gt-occluder-depth",
        "--ground-truth-occluder-depth",
        dest="gt_occluder_depth",
        required=True,
        type=Path,
    )
    parser.add_argument("--candidate-occluder-mask", required=True, type=Path)
    parser.add_argument("--candidate-occluder-depth", required=True, type=Path)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worst-frame-count", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = evaluate_files(
        robot_mask_path=args.robot_mask,
        robot_depth_path=args.robot_depth,
        gt_occluder_mask_path=args.gt_occluder_mask,
        gt_occluder_depth_path=args.gt_occluder_depth,
        candidate_occluder_mask_path=args.candidate_occluder_mask,
        candidate_occluder_depth_path=args.candidate_occluder_depth,
        output_path=args.output,
        sequence_id=args.sequence_id,
        candidate_name=args.candidate_name,
        worst_frame_count=args.worst_frame_count,
        overwrite=args.overwrite,
    )
    visibility = result["metrics"]["visibility"]["micro"]
    print(
        f"Evaluated {result['candidate_name']} on {result['sequence_id']}: "
        f"visible IoU={visibility['visible_iou']:.6f}, "
        f"decision errors={visibility['decision_disagreement']['count']} -> "
        f"{args.output.expanduser().resolve()}"
    )


if __name__ == "__main__":
    main()
