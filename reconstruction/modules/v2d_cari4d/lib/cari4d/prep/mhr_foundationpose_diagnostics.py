from __future__ import annotations

from typing import Any, Mapping

import numpy as np


FOUNDATIONPOSE_SELECTION_REVISION = "foundationpose-state-machine-v8-no-internal-symmetry-pending-confirmation-identity-cluster-score-ordered-retry-gt-first-rotation-oracle"
FOUNDATIONPOSE_SELECTION_FIELDS = (
    "fp_tracking_state",
    "fp_confirmed_pose_source",
    "fp_pending_pose_source",
    "fp_emitted_pose_source",
    "fp_gap_length",
    "fp_observed_visible_pixels",
    "fp_rendered_visible_pixels",
    "fp_rendered_visibility",
    "fp_full_image_iou",
    "fp_tolerant_precision",
    "fp_tolerant_recall",
    "fp_centroid_distance",
    "fp_confirmation_count",
    "fp_rejection_mask",
    "fp_reliable",
    "fp_reacquisition_latency",
)
FOUNDATIONPOSE_SELECTION_H5_PATHS = {key: f"object/foundationpose/{key[3:] if key.startswith('fp_') else key}" for key in FOUNDATIONPOSE_SELECTION_FIELDS}
FOUNDATIONPOSE_SELECTION_DTYPES = {
    "fp_tracking_state": np.dtype("uint8"),
    "fp_confirmed_pose_source": np.dtype("uint8"),
    "fp_pending_pose_source": np.dtype("uint8"),
    "fp_emitted_pose_source": np.dtype("uint8"),
    "fp_gap_length": np.dtype("uint16"),
    "fp_observed_visible_pixels": np.dtype("int32"),
    "fp_rendered_visible_pixels": np.dtype("int32"),
    "fp_rendered_visibility": np.dtype("float32"),
    "fp_full_image_iou": np.dtype("float32"),
    "fp_tolerant_precision": np.dtype("float32"),
    "fp_tolerant_recall": np.dtype("float32"),
    "fp_centroid_distance": np.dtype("float32"),
    "fp_confirmation_count": np.dtype("uint8"),
    "fp_rejection_mask": np.dtype("uint32"),
    "fp_reliable": np.dtype("bool"),
    "fp_reacquisition_latency": np.dtype("uint16"),
}
FOUNDATIONPOSE_REACQUISITION_LATENCY_UNKNOWN = np.iinfo(np.uint16).max


def validate_foundationpose_selection_metadata(metadata: Mapping[str, Any], label: str, camera_order: list[int] | None = None) -> None:
    if metadata.get("foundationpose_selection_revision") != FOUNDATIONPOSE_SELECTION_REVISION:
        raise ValueError(f"{label} has unsupported FoundationPose selection revision: {metadata.get('foundationpose_selection_revision')}")
    if metadata.get("foundationpose_selection_uses_depth") is not True:
        raise ValueError(f"{label}/foundationpose_selection_uses_depth must be true")
    if metadata.get("foundationpose_selection_uses_synchronized_cameras") is not False:
        raise ValueError(f"{label}/foundationpose_selection_uses_synchronized_cameras must be false")
    if not isinstance(metadata.get("foundationpose_selection_uses_gt_pose"), bool):
        raise ValueError(f"{label}/foundationpose_selection_uses_gt_pose must be Boolean")
    if not isinstance(metadata.get("first_usable_frame_gt_rotation_oracle"), bool):
        raise ValueError(f"{label}/first_usable_frame_gt_rotation_oracle must be Boolean")
    if metadata["foundationpose_selection_uses_gt_pose"] != metadata["first_usable_frame_gt_rotation_oracle"]:
        raise ValueError(f"{label} GT-pose selection metadata is inconsistent")
    if metadata.get("foundationpose_selection_uses_foundationpose_score") is not True:
        raise ValueError(f"{label}/foundationpose_selection_uses_foundationpose_score must be true")
    if metadata.get("foundationpose_candidate_generation_uses_object_symmetry") is not False:
        raise ValueError(f"{label}/foundationpose_candidate_generation_uses_object_symmetry must be false")
    if metadata.get("normal_cluster_symmetry_mode") != "identity_only":
        raise ValueError(f"{label}/normal_cluster_symmetry_mode must be identity_only")
    if metadata.get("retry_cluster_symmetry_mode") != "identity_only":
        raise ValueError(f"{label}/retry_cluster_symmetry_mode must be identity_only")
    if camera_order is not None and [int(item) for item in metadata.get("foundationpose_selection_camera_order", [])] != [int(item) for item in camera_order]:
        raise ValueError(f"{label} FoundationPose selection camera order differs: {metadata.get('foundationpose_selection_camera_order')} != {camera_order}")


def validate_foundationpose_selection_arrays(data: Mapping[str, Any], expected_shape: tuple[int, ...], label: str) -> dict[str, np.ndarray]:
    missing = [key for key in FOUNDATIONPOSE_SELECTION_FIELDS if key not in data]
    if missing:
        raise ValueError(f"{label} is missing FoundationPose selection fields: {missing}")
    validated: dict[str, np.ndarray] = {}
    for key in FOUNDATIONPOSE_SELECTION_FIELDS:
        value = np.asarray(data[key])
        if value.shape != expected_shape:
            raise ValueError(f"{label}/{key} must have shape {expected_shape}, got {value.shape}")
        expected_dtype = FOUNDATIONPOSE_SELECTION_DTYPES[key]
        if value.dtype != expected_dtype:
            raise TypeError(f"{label}/{key} must be {expected_dtype}, got {value.dtype}")
        if np.issubdtype(value.dtype, np.floating) and np.isinf(value).any():
            raise ValueError(f"{label}/{key} contains infinity")
        validated[key] = value
    for key in ("fp_observed_visible_pixels", "fp_rendered_visible_pixels"):
        if validated[key].size and int(validated[key].min()) < 0:
            raise ValueError(f"{label}/{key} contains a negative count")
    for key in ("fp_rendered_visibility", "fp_full_image_iou", "fp_tolerant_precision", "fp_tolerant_recall"):
        finite = np.isfinite(validated[key])
        if np.any((validated[key][finite] < 0.0) | (validated[key][finite] > 1.0)):
            raise ValueError(f"{label}/{key} contains a finite value outside [0,1]")
    return validated
