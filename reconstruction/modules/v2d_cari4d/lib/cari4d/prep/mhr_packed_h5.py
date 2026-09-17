from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from lib_mhr.object_symmetry import OBJECT_SYMMETRY_MODE_FINITE, OBJECT_SYMMETRY_MODE_FULL_SO3, OBJECT_SYMMETRY_MODE_NAMES, OBJECT_SYMMETRY_REVISION, validate_object_symmetry_metadata
from lib_mhr.object_pose_frame import resolve_object_pose_frame, stamp_object_pose_frame_metadata
from lib_mhr.camera_conventions import validate_mhr_init_root_metadata, validate_mhr_init_translation_metadata
from lib_mhr.geometry_provider import MHR_GEOMETRY_AUTHORITY_PARAMETER
from lib_mhr.hand_surface_contact import MHR_CONTACT_THRESHOLD_M, MHR_HAND_ORDER, MHR_HAND_SAMPLE_COUNT, MHR_HAND_SURFACE_CONTACT_PENDING_REVISION, MHR_HAND_SURFACE_CONTACT_REVISION, MHR_HAND_VERTEX_COUNT
from prep.mhr_geometry_certificate import MHR_GEOMETRY_CERTIFICATE_TOLERANCE_M, MHR_RECONSTRUCTIBLE_GEOMETRY_FIELDS, certify_parameter_authoritative_packed, stamp_parameter_authoritative_packed, validate_parameter_authoritative_certificate
from prep.mhr_foundationpose_diagnostics import FOUNDATIONPOSE_SELECTION_FIELDS, FOUNDATIONPOSE_SELECTION_H5_PATHS, FOUNDATIONPOSE_SELECTION_REVISION, validate_foundationpose_selection_arrays, validate_foundationpose_selection_metadata
from prep.mhr_foundationpose_training_tiers import FOUNDATIONPOSE_TRAINING_OBJECT_INITIALIZATION_SCHEMA, FOUNDATIONPOSE_TRAINING_TIER_FIELDS, FOUNDATIONPOSE_TRAINING_TIER_H5_PATHS, FOUNDATIONPOSE_TRAINING_TIER_REVISION, validate_foundationpose_training_tier_arrays, validate_foundationpose_training_tier_metadata
from prep.mhr_gt_object_visibility import GT_OBJECT_VISIBILITY_FIELDS, GT_OBJECT_VISIBILITY_H5_PATHS, GT_OBJECT_VISIBILITY_REVISION, validate_gt_object_visibility_arrays, validate_gt_object_visibility_metadata


SCHEMA = "cari4d.mhr_packed.v1"
TRAINING_TIER_SCHEMA = "cari4d.mhr_packed.foundationpose_training_tiers.v2"
PARAMETER_AUTHORITATIVE_SCHEMA = "cari4d.mhr_packed.parameter_authoritative.v2"
TRAINING_TIER_PARAMETER_AUTHORITATIVE_SCHEMA = "cari4d.mhr_packed.foundationpose_training_tiers.parameter_authoritative.v3"
SUPPORTED_SCHEMAS = frozenset((SCHEMA, TRAINING_TIER_SCHEMA, PARAMETER_AUTHORITATIVE_SCHEMA, TRAINING_TIER_PARAMETER_AUTHORITATIVE_SCHEMA))
TIER_SCHEMAS = frozenset((TRAINING_TIER_SCHEMA, TRAINING_TIER_PARAMETER_AUTHORITATIVE_SCHEMA))
PARAMETER_AUTHORITATIVE_SCHEMAS = frozenset((PARAMETER_AUTHORITATIVE_SCHEMA, TRAINING_TIER_PARAMETER_AUTHORITATIVE_SCHEMA))
TARGET_CHUNK_BYTES = 2 * 1024 * 1024
SMALL_DATASET_BYTES = 4096
PARAM_KEYS = ("mhr_global_rot6d", "mhr_trans", "mhr_body_pose_cont", "mhr_hand", "mhr_shape", "mhr_scale", "mhr_face")
GEOMETRY_KEYS = ("mhr_vertices", "mhr_joints", "mhr_keypoints")
PERSISTED_GEOMETRY_KEYS = tuple(key for key in GEOMETRY_KEYS if key not in MHR_RECONSTRUCTIBLE_GEOMETRY_FIELDS)
OPTIONAL_AUX_KEYS = ("mhr_joint_global_rots", "mhr_neutral_height")
QUALITY_FIELDS = frozenset(("official_fit_error_cm", "canonical_projection_error_cm", "canonical_source_error_cm"))
POSE_VALIDITY_FIELDS = ("human_pose_valid_mask", "object_pose_valid_mask")
ROOT_FIELDS = frozenset(("body_model", "seq_name", "frames", "kids", "gt", "init", "faces", "frame_valid_mask", *POSE_VALIDITY_FIELDS, "obj_rot", "obj_t", "obj_rot_gt", "obj_t_gt", "obj_rot_init", "obj_t_init", "obj_symmetry_tfs", "obj_symmetry_mode", "obj_symmetry_center", "mhr_contact_dist_gt", "mhr_contact_closest_points_gt", "quality", "metadata") + GT_OBJECT_VISIBILITY_FIELDS + FOUNDATIONPOSE_SELECTION_FIELDS + FOUNDATIONPOSE_TRAINING_TIER_FIELDS)
MHR_FIELDS = frozenset(("body_model", "frames", "kids", "metadata") + PARAM_KEYS + PERSISTED_GEOMETRY_KEYS + OPTIONAL_AUX_KEYS)
BASE_REQUIRED_ROOT_FIELDS = frozenset(("body_model", "seq_name", "frames", "kids", "gt", "init", "faces", "frame_valid_mask", "obj_rot_gt", "obj_t_gt", "obj_symmetry_tfs", "metadata"))
LEGACY_OBJECT_INIT_ROOT_FIELDS = frozenset(("obj_rot_init", "obj_t_init"))
REQUIRED_MHR_FIELDS = frozenset(PARAM_KEYS + PERSISTED_GEOMETRY_KEYS)
PARAMETER_AUTHORITATIVE_REQUIRED_MHR_FIELDS = REQUIRED_MHR_FIELDS
BASE_CANONICAL_REQUIRED_PATHS = frozenset(("frames", "kids", "metadata_json", "faces", "frame_valid_mask", "object/rot_gt", "object/trans_gt", "object/symmetry_transforms", "object/symmetry_mode", "object/symmetry_center") + tuple(f"{group}/{key}" for group in ("gt", "init") for key in PARAM_KEYS + PERSISTED_GEOMETRY_KEYS))
CONTACT_ROOT_FIELDS = ("mhr_contact_dist_gt", "mhr_contact_closest_points_gt")
CONTACT_REQUIRED_PATHS = frozenset(("contact/distance_gt", "contact/closest_points_gt"))
CACHED_VERTEX_PATHS = frozenset(f"{group}/mhr_vertices" for group in ("gt", "init"))
LEGACY_OBJECT_INIT_PATHS = frozenset(("object/rot_init", "object/trans_init"))
CANONICAL_REQUIRED_PATHS = BASE_CANONICAL_REQUIRED_PATHS | CONTACT_REQUIRED_PATHS | LEGACY_OBJECT_INIT_PATHS
FOUNDATIONPOSE_TRACKING_METADATA_PREFIXES = ("foundationpose_selection_", "foundationpose_candidate_generation_", "reacquisition_", "pending_consistency_")
FOUNDATIONPOSE_TRACKING_METADATA_KEYS = frozenset(("candidate_screen_max_side", "first_usable_frame_gt_rotation_oracle", "gt_rotation_oracle_frame_indices", "gt_rotation_oracle_frames", "normal_cluster_symmetry_mode", "retry_cluster_symmetry_mode", "selection_pipeline", "threshold_relaxation", "unrestricted_candidate_acceptance", "translation_base_threshold_m", "translation_history_size", "translation_history_multiplier"))


def canonical_required_paths_for_schema(schema: str) -> frozenset[str]:
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"Unsupported packed H5 schema: {schema}")
    paths = set(BASE_CANONICAL_REQUIRED_PATHS | CONTACT_REQUIRED_PATHS)
    paths.update(FOUNDATIONPOSE_TRAINING_TIER_H5_PATHS.values() if schema in TIER_SCHEMAS else LEGACY_OBJECT_INIT_PATHS)
    return frozenset(paths)


def foundationpose_tracking_metadata_keys(metadata: Mapping[str, Any]) -> list[str]:
    return sorted(key for key in metadata if key.startswith(FOUNDATIONPOSE_TRACKING_METADATA_PREFIXES) or key in FOUNDATIONPOSE_TRACKING_METADATA_KEYS)


def storage_options(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value)
    if value.ndim == 0 or value.nbytes < SMALL_DATASET_BYTES:
        return {}
    return {"compression": "lzf", "shuffle": value.dtype.itemsize > 1}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping, got {type(value).__name__}")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {unknown}")


def _require_fields(mapping: Mapping[str, Any], required: frozenset[str], label: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"Missing required {label} fields: {missing}")


def _require_array(mapping: Mapping[str, Any], key: str, label: str) -> np.ndarray:
    if key not in mapping:
        raise KeyError(f"{label} is missing required field {key}")
    value = np.asarray(mapping[key])
    if value.dtype == object:
        raise TypeError(f"{label}/{key} has unsupported object dtype")
    return value


def _require_float32(value: np.ndarray, label: str) -> np.ndarray:
    if value.dtype != np.dtype("float32"):
        raise TypeError(f"{label} must be float32 for lossless canonical storage, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    return value


def _safe_int_cast(value: np.ndarray, dtype: np.dtype, label: str) -> np.ndarray:
    if not np.issubdtype(value.dtype, np.integer):
        raise TypeError(f"{label} must be integer, got {value.dtype}")
    info = np.iinfo(dtype)
    if value.size and (int(value.min()) < info.min or int(value.max()) > info.max):
        raise OverflowError(f"{label} cannot be represented as {dtype}")
    return value.astype(dtype, copy=False)


def _json_value(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item(), label)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise TypeError(f"{label} contains an object-dtype array")
        return _json_value(value.tolist(), label)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, f"{label}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{label}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{label} contains unsupported value type {type(value).__name__}")


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(_json_value(value, "metadata"), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_repeated_mhr_fields(group: Mapping[str, Any], frames: list[str], kids: list[int], label: str) -> None:
    if "body_model" in group and str(group["body_model"]) != "mhr":
        raise ValueError(f"{label}/body_model must be mhr, got {group['body_model']}")
    if "frames" in group and [str(item) for item in group["frames"]] != frames:
        raise ValueError(f"{label}/frames does not match top-level frames")
    if "kids" in group and [int(item) for item in group["kids"]] != kids:
        raise ValueError(f"{label}/kids does not match top-level kids")
    if "metadata" in group:
        _require_mapping(group["metadata"], f"{label}/metadata")


def _check_shape(value: np.ndarray, expected_prefix: tuple[int, ...], label: str) -> None:
    if value.shape[:len(expected_prefix)] != expected_prefix:
        raise ValueError(f"{label} expected leading shape {expected_prefix}, got {value.shape}")


def _binary_frame_mask(packed: Mapping[str, Any], key: str, frame_count: int) -> np.ndarray:
    mask = _require_array(packed, key, "packed")
    if mask.shape != (frame_count,):
        raise ValueError(f"{key} must have shape ({frame_count},), got {mask.shape}")
    if mask.dtype != np.dtype("bool"):
        if not np.isin(mask, (0, 1)).all():
            raise ValueError(f"{key} contains values other than 0/1")
        mask = mask.astype(bool)
    return mask


def collect_packed_datasets(packed: Mapping[str, Any], *, allow_pending_contacts: bool = False) -> dict[str, Any]:
    packed = _require_mapping(packed, "packed")
    _reject_unknown(packed, ROOT_FIELDS, "packed")
    _require_fields(packed, BASE_REQUIRED_ROOT_FIELDS, "packed")
    metadata = stamp_object_pose_frame_metadata(_require_mapping(packed["metadata"], "metadata"), assume_aligned_without_mesh=True)
    validate_mhr_init_root_metadata(metadata, "packed metadata")
    validate_mhr_init_translation_metadata(metadata, "packed metadata")
    contact_fields_present = [key for key in CONTACT_ROOT_FIELDS if key in packed]
    pending_contacts = metadata.get("mhr_contact_revision") == MHR_HAND_SURFACE_CONTACT_PENDING_REVISION
    if pending_contacts:
        validate_mhr_pending_contact_metadata(metadata, "packed metadata")
        if not allow_pending_contacts:
            raise ValueError("Pending hand-surface contacts are valid only inside the pack-to-contact production boundary")
        if contact_fields_present:
            raise ValueError(f"Pending hand-surface contacts must not contain contact arrays: {contact_fields_present}")
    elif len(contact_fields_present) != len(CONTACT_ROOT_FIELDS):
        raise ValueError(f"Packed contact arrays must contain both {CONTACT_ROOT_FIELDS}, got {contact_fields_present}")
    tier_fields_present = [key for key in FOUNDATIONPOSE_TRAINING_TIER_FIELDS if key in packed]
    tier_revision = metadata.get("foundationpose_training_tier_revision")
    tier_schema = bool(tier_fields_present or tier_revision is not None)
    parameter_authoritative = metadata.get("mhr_geometry_authority") == MHR_GEOMETRY_AUTHORITY_PARAMETER
    schema = TRAINING_TIER_PARAMETER_AUTHORITATIVE_SCHEMA if tier_schema and parameter_authoritative else TRAINING_TIER_SCHEMA if tier_schema else PARAMETER_AUTHORITATIVE_SCHEMA if parameter_authoritative else SCHEMA
    if tier_schema:
        legacy_object_init_fields = sorted(LEGACY_OBJECT_INIT_ROOT_FIELDS & set(packed))
        if legacy_object_init_fields:
            raise ValueError(f"Tier-only packed input must not contain duplicate legacy object initialization fields: {legacy_object_init_fields}")
        selection_fields_present = [key for key in FOUNDATIONPOSE_SELECTION_FIELDS if key in packed]
        if selection_fields_present:
            raise ValueError(f"Tier-only packed input must not contain FoundationPose temporal-selection fields: {selection_fields_present}")
        tracking_metadata = foundationpose_tracking_metadata_keys(metadata)
        if tracking_metadata:
            raise ValueError(f"Tier-only packed metadata must not contain FoundationPose temporal-tracking metadata: {tracking_metadata}")
        if metadata.get("foundationpose_training_object_initialization_schema") != FOUNDATIONPOSE_TRAINING_OBJECT_INITIALIZATION_SCHEMA:
            raise ValueError(f"Tier-only packed metadata must declare foundationpose_training_object_initialization_schema={FOUNDATIONPOSE_TRAINING_OBJECT_INITIALIZATION_SCHEMA!r}")
    else:
        _require_fields(packed, LEGACY_OBJECT_INIT_ROOT_FIELDS, "legacy packed")
    if str(packed["body_model"]) != "mhr":
        raise ValueError(f"body_model must be mhr, got {packed['body_model']}")
    frames = [str(item) for item in packed["frames"]]
    kids_raw = np.asarray(packed["kids"])
    kids = _safe_int_cast(kids_raw, np.dtype("int16"), "kids").tolist()
    if not frames:
        raise ValueError("frames must be non-empty")
    if len(set(frames)) != len(frames):
        raise ValueError("frames contains duplicate names")
    if not kids:
        raise ValueError("kids must be non-empty")
    if len(set(kids)) != len(kids):
        raise ValueError("kids contains duplicate camera ids")
    frame_count = len(frames)
    camera_count = len(kids)
    datasets: dict[str, np.ndarray] = {}
    for group_name in ("gt", "init"):
        group = _require_mapping(packed[group_name], group_name)
        _reject_unknown(group, MHR_FIELDS, group_name)
        _require_fields(group, PARAMETER_AUTHORITATIVE_REQUIRED_MHR_FIELDS if parameter_authoritative else REQUIRED_MHR_FIELDS, group_name)
        _validate_repeated_mhr_fields(group, frames, kids, group_name)
        for key in PARAM_KEYS + PERSISTED_GEOMETRY_KEYS + OPTIONAL_AUX_KEYS:
            if key not in group:
                continue
            value = _require_float32(_require_array(group, key, group_name), f"{group_name}/{key}")
            _check_shape(value, (frame_count, camera_count) if group_name == "init" else (frame_count,), f"{group_name}/{key}")
            datasets[f"{group_name}/{key}"] = value
    if parameter_authoritative:
        validate_parameter_authoritative_certificate(metadata, _require_mapping(packed["gt"], "gt"), _require_mapping(packed["init"], "init"))
    faces = _require_array(packed, "faces", "packed")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape [F,3], got {faces.shape}")
    datasets["faces"] = _safe_int_cast(faces, np.dtype("int32"), "faces")
    frame_valid_mask = _binary_frame_mask(packed, "frame_valid_mask", frame_count)
    datasets["frame_valid_mask"] = frame_valid_mask
    validity_fields_present = [key for key in POSE_VALIDITY_FIELDS if key in packed]
    if validity_fields_present and len(validity_fields_present) != len(POSE_VALIDITY_FIELDS):
        raise ValueError(f"Packed pose validity must contain both {POSE_VALIDITY_FIELDS}, got {validity_fields_present}")
    if validity_fields_present:
        human_pose_valid = _binary_frame_mask(packed, "human_pose_valid_mask", frame_count)
        object_pose_valid = _binary_frame_mask(packed, "object_pose_valid_mask", frame_count)
        if not np.array_equal(frame_valid_mask, human_pose_valid & object_pose_valid):
            raise ValueError("frame_valid_mask must equal human_pose_valid_mask & object_pose_valid_mask")
        datasets["human_pose_valid_mask"] = human_pose_valid
        datasets["object_pose_valid_mask"] = object_pose_valid
    object_specs = {"object/rot_gt": ("obj_rot_gt", (frame_count, 3, 3)), "object/trans_gt": ("obj_t_gt", (frame_count, 3)), "object/symmetry_transforms": ("obj_symmetry_tfs", None)}
    if not pending_contacts:
        object_specs.update({"contact/distance_gt": ("mhr_contact_dist_gt", (frame_count, 2)), "contact/closest_points_gt": ("mhr_contact_closest_points_gt", (frame_count, 2, 3))})
    if not tier_schema:
        object_specs.update({"object/rot_init": ("obj_rot_init", (frame_count, camera_count, 3, 3)), "object/trans_init": ("obj_t_init", (frame_count, camera_count, 3))})
    for path, (key, expected_shape) in object_specs.items():
        value = _require_float32(_require_array(packed, key, "packed"), key)
        if expected_shape is not None and value.shape != expected_shape:
            raise ValueError(f"{key} expected shape {expected_shape}, got {value.shape}")
        datasets[path] = value
    symmetry = datasets["object/symmetry_transforms"]
    if symmetry.ndim != 3 or symmetry.shape[0] < 1 or symmetry.shape[1:] != (4, 4):
        raise ValueError(f"obj_symmetry_tfs must have shape [S,4,4], got {symmetry.shape}")
    symmetry_mode = np.asarray(packed.get("obj_symmetry_mode", OBJECT_SYMMETRY_MODE_FINITE))
    if symmetry_mode.shape != () or not np.issubdtype(symmetry_mode.dtype, np.integer) or int(symmetry_mode) not in OBJECT_SYMMETRY_MODE_NAMES:
        raise ValueError(f"obj_symmetry_mode must be a supported integer scalar, got {symmetry_mode.shape} {symmetry_mode.dtype}")
    symmetry_center = _require_float32(np.asarray(packed.get("obj_symmetry_center", np.zeros(3, dtype=np.float32))), "obj_symmetry_center")
    if symmetry_center.shape != (3,):
        raise ValueError(f"obj_symmetry_center must have shape [3], got {symmetry_center.shape}")
    if int(symmetry_mode) == OBJECT_SYMMETRY_MODE_FULL_SO3 and (len(symmetry) != 1 or not np.allclose(symmetry[0], np.eye(4), rtol=0.0, atol=1e-6)):
        raise ValueError("Full SO(3) packed symmetry must use one identity finite representative")
    datasets["object/symmetry_mode"] = symmetry_mode.astype(np.uint8)
    datasets["object/symmetry_center"] = symmetry_center
    symmetry_metadata = {
        "object_symmetry_revision": OBJECT_SYMMETRY_REVISION,
        "object_symmetry_mode": OBJECT_SYMMETRY_MODE_NAMES[int(symmetry_mode)],
        "object_symmetry_center": symmetry_center.tolist(),
    }
    for key, expected in symmetry_metadata.items():
        if key in metadata and metadata[key] != expected:
            raise ValueError(f"Packed metadata {key} disagrees with canonical object symmetry data: metadata={metadata[key]!r}, expected={expected!r}")
        metadata[key] = expected
    for alias, canonical in (("obj_rot", "obj_rot_gt"), ("obj_t", "obj_t_gt")):
        if alias in packed and not np.array_equal(np.asarray(packed[alias]), np.asarray(packed[canonical])):
            raise ValueError(f"{alias} does not exactly match {canonical}")
    visibility_fields_present = [key for key in GT_OBJECT_VISIBILITY_FIELDS if key in packed]
    visibility_revision = metadata.get("gt_object_visibility_revision")
    if visibility_fields_present or visibility_revision is not None:
        validate_gt_object_visibility_metadata(metadata, "packed metadata", kids)
        visibility = validate_gt_object_visibility_arrays(packed, (frame_count, camera_count), "packed")
        for key, value in visibility.items():
            datasets[GT_OBJECT_VISIBILITY_H5_PATHS[key]] = value
    selection_fields_present = [key for key in FOUNDATIONPOSE_SELECTION_FIELDS if key in packed]
    selection_revision = metadata.get("foundationpose_selection_revision")
    if tier_schema and selection_revision is not None:
        raise ValueError("Tier-only packed metadata must not contain foundationpose_selection_revision")
    if not tier_schema and (selection_fields_present or selection_revision is not None):
        validate_foundationpose_selection_metadata(metadata, "packed metadata", kids)
        selection = validate_foundationpose_selection_arrays(packed, (frame_count, camera_count), "packed")
        for key, value in selection.items():
            datasets[FOUNDATIONPOSE_SELECTION_H5_PATHS[key]] = value
    if tier_schema:
        validate_foundationpose_training_tier_metadata(metadata, "packed metadata")
        tiers = validate_foundationpose_training_tier_arrays(packed, (frame_count, camera_count), "packed")
        expected_gt_valid = np.repeat(datasets.get("object_pose_valid_mask", frame_valid_mask)[:, None], camera_count, axis=1)
        if not np.array_equal(tiers["fp_training_gt_valid"], expected_gt_valid):
            raise ValueError("fp_training_gt_valid must equal object_pose_valid_mask for every camera")
        tier_dtypes = {
            "fp_training_tier_poses_world": np.float32, "fp_training_tier_valid": np.bool_, "fp_training_tier_rotation_error_deg": np.float32,
            "fp_training_tier_translation_error_m": np.float32, "fp_training_tier_translation_error_diameter": np.float32, "fp_training_tier_score": np.float32,
            "fp_training_tier_score_rank": np.int32, "fp_training_tier_source_id": np.int32, "fp_training_tier_symmetry_index": np.int16,
            "fp_training_top1_tier": np.uint8, "fp_training_gt_valid": np.bool_, "fp_training_reference_type": np.uint8, "fp_training_candidate_count": np.int32,
            "fp_training_imputed": np.bool_, "fp_training_donor_frame_index": np.int32, "fp_training_donor_frame_offset": np.int32, "fp_training_missing_reason": np.uint8,
        }
        for key, value in tiers.items():
            datasets[FOUNDATIONPOSE_TRAINING_TIER_H5_PATHS[key]] = np.asarray(value, dtype=tier_dtypes[key])
    if "quality" in packed:
        quality = _require_mapping(packed["quality"], "quality")
        _reject_unknown(quality, QUALITY_FIELDS, "quality")
        for key, raw_value in quality.items():
            value = _require_float32(np.asarray(raw_value), f"quality/{key}")
            if value.shape != (frame_count,):
                raise ValueError(f"quality/{key} expected shape ({frame_count},), got {value.shape}")
            datasets[f"quality/{key}"] = value
    validate_object_symmetry_metadata(metadata)
    resolve_object_pose_frame(metadata)
    return {"datasets": datasets, "frames": frames, "kids": kids, "sequence": str(packed["seq_name"]), "body_model": "mhr", "metadata_json": _json_text(metadata), "schema": schema}


def _is_view_specific(path: str, value: Any, frame_count: int, camera_count: int) -> bool:
    return value.ndim >= 2 and value.shape[:2] == (frame_count, camera_count) and (path.startswith("init/") or path in {"object/rot_init", "object/trans_init", *GT_OBJECT_VISIBILITY_H5_PATHS.values(), *FOUNDATIONPOSE_SELECTION_H5_PATHS.values(), *FOUNDATIONPOSE_TRAINING_TIER_H5_PATHS.values()})


def _chunk_shape(path: str, value: np.ndarray, frame_count: int, camera_count: int) -> tuple[int, ...] | bool | None:
    if value.ndim == 0 or value.size == 0:
        return None
    if value.shape[0] != frame_count:
        trailing_shape = value.shape[1:]
        bytes_per_item = max(1, int(np.prod(trailing_shape, dtype=np.int64)) * value.dtype.itemsize)
        item_chunk = min(value.shape[0], max(1, TARGET_CHUNK_BYTES // bytes_per_item))
        return (item_chunk, *trailing_shape)
    view_specific = _is_view_specific(path, value, frame_count, camera_count)
    trailing_shape = value.shape[2:] if view_specific else value.shape[1:]
    bytes_per_frame = max(1, int(np.prod(trailing_shape, dtype=np.int64)) * value.dtype.itemsize)
    frame_chunk = min(frame_count, 96, max(1, TARGET_CHUNK_BYTES // bytes_per_frame))
    return (frame_chunk, 1, *value.shape[2:]) if view_specific else (frame_chunk, *value.shape[1:])


def _create_numeric_dataset(handle: h5py.File, path: str, value: np.ndarray, frame_count: int, camera_count: int) -> None:
    group_path, name = path.rsplit("/", 1) if "/" in path else ("", path)
    group = handle.require_group(group_path) if group_path else handle
    options = storage_options(value)
    if options:
        options["chunks"] = _chunk_shape(path, value, frame_count, camera_count)
    group.create_dataset(name, data=value, **options)


def _decode_strings(value: np.ndarray) -> list[str]:
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in value]


def packed_metadata(handle: h5py.File) -> dict[str, Any]:
    if "metadata_json" not in handle:
        raise KeyError("Packed H5 is missing required dataset metadata_json")
    metadata_text = handle["metadata_json"][()]
    if isinstance(metadata_text, bytes):
        metadata_text = metadata_text.decode("utf-8")
    metadata = json.loads(str(metadata_text))
    if not isinstance(metadata, dict):
        raise TypeError(f"Packed H5 metadata_json must decode to an object, got {type(metadata).__name__}")
    return metadata


def validate_packed_mhr_init_root(handle: h5py.File, label: str) -> dict[str, Any]:
    metadata = packed_metadata(handle)
    root_metadata = validate_mhr_init_root_metadata(metadata, f"{label} metadata")
    translation_metadata = validate_mhr_init_translation_metadata(metadata, f"{label} metadata")
    for key, expected in {**root_metadata, **translation_metadata}.items():
        if str(handle.attrs.get(key, "")) != expected:
            raise ValueError(f"{label} attribute {key}={handle.attrs.get(key)!r}, expected {expected!r}")
    return metadata


def validate_mhr_hand_surface_contact_metadata(metadata: Mapping[str, Any], label: str, expected_revision: str = MHR_HAND_SURFACE_CONTACT_REVISION) -> dict[str, Any]:
    if expected_revision != MHR_HAND_SURFACE_CONTACT_REVISION:
        raise ValueError(f"Unsupported required MHR contact revision {expected_revision!r}")
    expected = {
        "mhr_contact_revision": expected_revision,
        "mhr_contact_distance_units": "meters",
        "mhr_contact_hand_order": list(MHR_HAND_ORDER),
        "mhr_contact_hand_vertex_count": [MHR_HAND_VERTEX_COUNT, MHR_HAND_VERTEX_COUNT],
        "mhr_contact_sample_count": MHR_HAND_SAMPLE_COUNT,
        "mhr_contact_threshold_m": MHR_CONTACT_THRESHOLD_M,
        "mhr_contact_closest_points_space": "world",
    }
    differing = {key: {"actual": metadata.get(key), "expected": value} for key, value in expected.items() if metadata.get(key) != value}
    if differing:
        raise ValueError(f"{label} does not satisfy the required hand-surface contact schema: {differing}")
    if "mhr_contact_wrist_order" in metadata:
        raise ValueError(f"{label} contains legacy wrist-contact metadata")
    return {key: metadata[key] for key in expected}


def validate_mhr_pending_contact_metadata(metadata: Mapping[str, Any], label: str) -> dict[str, Any]:
    if metadata.get("mhr_contact_revision") != MHR_HAND_SURFACE_CONTACT_PENDING_REVISION:
        raise ValueError(f"{label} must declare {MHR_HAND_SURFACE_CONTACT_PENDING_REVISION!r}")
    source = metadata.get("mhr_contact_source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{label} pending hand-surface contacts require a nonempty mhr_contact_source")
    stale = sorted(key for key in metadata if key.startswith("mhr_contact_") and key not in {"mhr_contact_revision", "mhr_contact_source"})
    if stale:
        raise ValueError(f"{label} pending hand-surface contacts contain stale contact metadata: {stale}")
    return {"mhr_contact_revision": MHR_HAND_SURFACE_CONTACT_PENDING_REVISION, "mhr_contact_source": source}


def validate_packed_mhr_pending_contacts(handle: h5py.File, label: str) -> dict[str, Any]:
    metadata = packed_metadata(handle)
    validated = validate_mhr_pending_contact_metadata(metadata, f"{label} metadata")
    present = sorted(path for path in CONTACT_REQUIRED_PATHS if path in handle)
    if present or "contact" in handle:
        raise ValueError(f"{label} pending hand-surface contacts must not contain contact datasets: {present}")
    return validated


def validate_packed_mhr_hand_surface_contacts(handle: h5py.File, label: str, expected_revision: str = MHR_HAND_SURFACE_CONTACT_REVISION) -> dict[str, Any]:
    metadata = packed_metadata(handle)
    validated = validate_mhr_hand_surface_contact_metadata(metadata, f"{label} metadata", expected_revision)
    frame_count = len(handle["frames"])
    expected = {"contact/distance_gt": ((frame_count, 2), np.dtype("float32")), "contact/closest_points_gt": ((frame_count, 2, 3), np.dtype("float32"))}
    for path, (shape, dtype) in expected.items():
        if path not in handle:
            raise KeyError(f"{label} is missing required hand-surface contact dataset {path}")
        dataset = handle[path]
        if dataset.shape != shape or dataset.dtype != dtype:
            raise ValueError(f"{label} {path} must have shape/dtype {shape}/{dtype}, got {dataset.shape}/{dataset.dtype}")
        values = dataset[()]
        if not np.isfinite(values).all() or path.endswith("distance_gt") and np.any(values < 0.0):
            raise ValueError(f"{label} {path} must contain finite values and nonnegative distances")
    return validated


def _validate_handle(handle: h5py.File, collected: Mapping[str, Any] | None, expected_source_sha256: str | None, require_complete: bool, *, allow_pending_contacts: bool = False) -> dict[str, Any]:
    schema = str(handle.attrs.get("schema", ""))
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"Unsupported packed H5 schema: {schema}")
    if require_complete and not bool(handle.attrs.get("complete", False)):
        raise ValueError("Packed H5 is not complete")
    if expected_source_sha256 is not None and str(handle.attrs.get("source_sha256", "")) != expected_source_sha256:
        raise ValueError(f"Packed H5 source SHA256 does not match expected {expected_source_sha256}")
    required_paths = set(BASE_CANONICAL_REQUIRED_PATHS)
    metadata_text = handle["metadata_json"][()]
    if isinstance(metadata_text, bytes):
        metadata_text = metadata_text.decode("utf-8")
    metadata = validate_packed_mhr_init_root(handle, "Packed H5")
    pending_contacts = metadata.get("mhr_contact_revision") == MHR_HAND_SURFACE_CONTACT_PENDING_REVISION
    if pending_contacts:
        if not allow_pending_contacts:
            raise ValueError("Packed H5 contains pending hand-surface contacts and cannot be published")
        validate_packed_mhr_pending_contacts(handle, "Packed H5")
    else:
        required_paths.update(CONTACT_REQUIRED_PATHS)
        validate_packed_mhr_hand_surface_contacts(handle, "Packed H5")
    tier_schema = schema in TIER_SCHEMAS
    parameter_authoritative = schema in PARAMETER_AUTHORITATIVE_SCHEMAS
    if parameter_authoritative:
        forbidden_vertices = sorted(path for path in CACHED_VERTEX_PATHS if path in handle)
        if forbidden_vertices:
            raise ValueError(f"Parameter-authoritative packed H5 contains forbidden cached vertices: {forbidden_vertices}")
        validate_parameter_authoritative_certificate(metadata, {key: handle[f"gt/{key}"][()] for key in PARAM_KEYS}, {key: handle[f"init/{key}"][()] for key in PARAM_KEYS})
    if tier_schema:
        forbidden_paths = sorted(path for path in (*LEGACY_OBJECT_INIT_PATHS, *FOUNDATIONPOSE_SELECTION_H5_PATHS.values()) if path in handle)
        if "object/foundationpose" in handle and "object/foundationpose" not in forbidden_paths:
            forbidden_paths.append("object/foundationpose")
        if forbidden_paths:
            raise ValueError(f"Tier-only packed H5 contains forbidden legacy FoundationPose datasets: {forbidden_paths}")
        tracking_metadata = foundationpose_tracking_metadata_keys(metadata)
        if tracking_metadata:
            raise ValueError(f"Tier-only packed H5 metadata contains forbidden FoundationPose temporal-tracking metadata: {tracking_metadata}")
        validate_foundationpose_training_tier_metadata(metadata, "packed H5 metadata")
        required_paths.update(FOUNDATIONPOSE_TRAINING_TIER_H5_PATHS.values())
    else:
        required_paths.update(LEGACY_OBJECT_INIT_PATHS)
        stale_tier_paths = sorted(path for path in FOUNDATIONPOSE_TRAINING_TIER_H5_PATHS.values() if path in handle)
        if metadata.get("foundationpose_training_tier_revision") is not None or stale_tier_paths:
            raise ValueError(f"Legacy packed H5 must not contain FoundationPose training tiers; regenerate with schema {TRAINING_TIER_SCHEMA}: {stale_tier_paths}")
    visibility_revision = metadata.get("gt_object_visibility_revision")
    visibility_paths_present = [path for path in GT_OBJECT_VISIBILITY_H5_PATHS.values() if path in handle]
    if visibility_revision is not None or visibility_paths_present:
        validate_gt_object_visibility_metadata(metadata, "packed H5 metadata", [int(kid) for kid in handle["kids"][()]])
        required_paths.update(GT_OBJECT_VISIBILITY_H5_PATHS.values())
    selection_revision = metadata.get("foundationpose_selection_revision")
    selection_paths_present = [path for path in FOUNDATIONPOSE_SELECTION_H5_PATHS.values() if path in handle]
    if tier_schema and (selection_revision is not None or selection_paths_present):
        raise ValueError("Tier-only packed H5 must not contain FoundationPose temporal-selection diagnostics")
    if not tier_schema and (selection_revision is not None or selection_paths_present):
        validate_foundationpose_selection_metadata(metadata, "packed H5 metadata", [int(kid) for kid in handle["kids"][()]])
        required_paths.update(FOUNDATIONPOSE_SELECTION_H5_PATHS.values())
    tier_revision = metadata.get("foundationpose_training_tier_revision")
    tier_paths_present = [path for path in FOUNDATIONPOSE_TRAINING_TIER_H5_PATHS.values() if path in handle]
    if tier_schema and tier_revision != FOUNDATIONPOSE_TRAINING_TIER_REVISION:
        raise ValueError(f"Tier-only packed H5 has stale FoundationPose training-tier revision: {tier_revision}")
    checked_values = 0
    if collected is not None:
        required_paths.update(collected["datasets"])
        if schema != collected["schema"]:
            raise ValueError(f"Packed H5 schema does not match source: stored={schema}, source={collected['schema']}")
        if str(handle.attrs["sequence"]) != collected["sequence"] or str(handle.attrs["body_model"]) != collected["body_model"]:
            raise ValueError("Packed H5 root attributes do not match source")
        if _decode_strings(handle["frames"][()]) != collected["frames"]:
            raise ValueError("Packed H5 frames do not match source")
        if not np.array_equal(handle["kids"][()], np.asarray(collected["kids"], dtype=np.int16)):
            raise ValueError("Packed H5 kids do not match source")
        if str(metadata_text) != collected["metadata_json"]:
            raise ValueError("Packed H5 metadata does not match source")
        frame_count = len(collected["frames"])
        camera_count = len(collected["kids"])
        for path, expected in collected["datasets"].items():
            dataset = handle[path]
            if dataset.shape != expected.shape or dataset.dtype != expected.dtype:
                raise ValueError(f"{path} shape/dtype mismatch: source={expected.shape}/{expected.dtype}, stored={dataset.shape}/{dataset.dtype}")
            options = storage_options(expected)
            if options:
                if dataset.compression != "lzf" or bool(dataset.shuffle) != bool(options["shuffle"]):
                    raise ValueError(f"{path} storage policy mismatch: compression={dataset.compression}, shuffle={dataset.shuffle}")
                if dataset.chunks != _chunk_shape(path, expected, frame_count, camera_count):
                    raise ValueError(f"{path} chunk mismatch: expected {_chunk_shape(path, expected, frame_count, camera_count)}, got {dataset.chunks}")
            elif dataset.compression is not None:
                raise ValueError(f"{path} must be uncompressed because it is smaller than {SMALL_DATASET_BYTES} bytes")
            actual = dataset[()]
            values_equal = np.array_equal(actual, expected, equal_nan=True) if np.issubdtype(expected.dtype, np.floating) else np.array_equal(actual, expected)
            if not values_equal:
                raise ValueError(f"{path} values do not exactly match source")
            checked_values += int(expected.size)
    for path in required_paths:
        if path not in handle:
            raise ValueError(f"Packed H5 is missing required dataset {path}")
    symmetry_mode = np.asarray(handle["object/symmetry_mode"][()])
    symmetry_center = np.asarray(handle["object/symmetry_center"][()], dtype=np.float32)
    symmetry_transforms = np.asarray(handle["object/symmetry_transforms"][()], dtype=np.float32)
    if symmetry_mode.shape != () or not np.issubdtype(symmetry_mode.dtype, np.integer) or int(symmetry_mode) not in OBJECT_SYMMETRY_MODE_NAMES:
        raise ValueError(f"Packed H5 object/symmetry_mode is invalid: {symmetry_mode.shape} {symmetry_mode.dtype}")
    if symmetry_center.shape != (3,) or not np.isfinite(symmetry_center).all():
        raise ValueError(f"Packed H5 object/symmetry_center must be finite with shape [3], got {symmetry_center.shape}")
    if metadata.get("object_symmetry_revision") != OBJECT_SYMMETRY_REVISION or metadata.get("object_symmetry_mode") != OBJECT_SYMMETRY_MODE_NAMES[int(symmetry_mode)] or not np.array_equal(np.asarray(metadata.get("object_symmetry_center"), dtype=np.float32), symmetry_center):
        raise ValueError("Packed H5 object symmetry datasets disagree with metadata")
    if int(symmetry_mode) == OBJECT_SYMMETRY_MODE_FULL_SO3 and (len(symmetry_transforms) != 1 or not np.allclose(symmetry_transforms[0], np.eye(4), rtol=0.0, atol=1e-6)):
        raise ValueError("Packed H5 full SO(3) symmetry must use one identity finite representative")
    validate_object_symmetry_metadata(metadata)
    if visibility_revision == GT_OBJECT_VISIBILITY_REVISION:
        frame_count = len(handle["frames"])
        camera_count = len(handle["kids"])
        validate_gt_object_visibility_arrays({key: handle[path][()] for key, path in GT_OBJECT_VISIBILITY_H5_PATHS.items()}, (frame_count, camera_count), "packed H5")
    if not tier_schema and selection_revision == FOUNDATIONPOSE_SELECTION_REVISION:
        frame_count = len(handle["frames"])
        camera_count = len(handle["kids"])
        validate_foundationpose_selection_arrays({key: handle[path][()] for key, path in FOUNDATIONPOSE_SELECTION_H5_PATHS.items()}, (frame_count, camera_count), "packed H5")
    if tier_schema:
        frame_count = len(handle["frames"])
        camera_count = len(handle["kids"])
        tiers = validate_foundationpose_training_tier_arrays({key: handle[path][()] for key, path in FOUNDATIONPOSE_TRAINING_TIER_H5_PATHS.items()}, (frame_count, camera_count), "packed H5")
        object_pose_valid = np.asarray(handle["object_pose_valid_mask"][()] if "object_pose_valid_mask" in handle else handle["frame_valid_mask"][()], dtype=bool)
        expected_gt_valid = np.repeat(object_pose_valid[:, None], camera_count, axis=1)
        if not np.array_equal(tiers["fp_training_gt_valid"], expected_gt_valid):
            raise ValueError("packed H5 fp_training_gt_valid must equal object_pose_valid_mask for every camera")
    return {"sequence": str(handle.attrs["sequence"]), "schema": schema, "source_sha256": str(handle.attrs.get("source_sha256", "")), "dataset_count": len(required_paths), "validated_value_count": checked_values, "gt_object_visibility_revision": visibility_revision, "foundationpose_selection_revision": selection_revision, "foundationpose_training_tier_revision": tier_revision, "mhr_geometry_authority": metadata.get("mhr_geometry_authority", "cached_geometry"), "mhr_contact_revision": metadata.get("mhr_contact_revision")}


def write_packed_h5(packed: Mapping[str, Any], output_path: str | Path, *, source_path: str | Path | None = None, source_sha256: str | None = None, source_bytes: int | None = None, source_mtime_ns: int | None = None, allow_pending_contacts: bool = False) -> dict[str, Any]:
    collected = collect_packed_datasets(packed, allow_pending_contacts=allow_pending_contacts)
    if collected["schema"] not in PARAMETER_AUTHORITATIVE_SCHEMAS:
        raise ValueError("MHR packed publication requires parameter-authoritative data; use write_parameter_authoritative_packed_h5 with an MHR decoder")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(str(output_path) + ".tmp")
    temporary_path.unlink(missing_ok=True)
    with h5py.File(temporary_path, "w") as handle:
        handle.attrs["schema"] = collected["schema"]
        handle.attrs["complete"] = False
        handle.attrs["sequence"] = collected["sequence"]
        handle.attrs["body_model"] = collected["body_model"]
        handle.attrs["codec"] = "lzf"
        metadata = json.loads(collected["metadata_json"])
        frame_metadata = {**validate_mhr_init_root_metadata(metadata, "packed metadata"), **validate_mhr_init_translation_metadata(metadata, "packed metadata")}
        for key, value in frame_metadata.items():
            handle.attrs[key] = value
        if source_path is not None:
            handle.attrs["source_path"] = str(Path(source_path).resolve())
        if source_sha256 is not None:
            handle.attrs["source_sha256"] = source_sha256
        if source_bytes is not None:
            handle.attrs["source_bytes"] = int(source_bytes)
        if source_mtime_ns is not None:
            handle.attrs["source_mtime_ns"] = int(source_mtime_ns)
        handle.create_dataset("frames", data=np.asarray(collected["frames"], dtype=h5py.string_dtype("utf-8")))
        handle.create_dataset("kids", data=np.asarray(collected["kids"], dtype=np.int16))
        handle.create_dataset("metadata_json", data=collected["metadata_json"], dtype=h5py.string_dtype("utf-8"))
        for path, value in collected["datasets"].items():
            _create_numeric_dataset(handle, path, value, len(collected["frames"]), len(collected["kids"]))
        handle.flush()
        _validate_handle(handle, collected, source_sha256, require_complete=False, allow_pending_contacts=allow_pending_contacts)
        handle.attrs.modify("complete", True)
        handle.flush()
    os.replace(temporary_path, output_path)
    report = validate_packed_h5(output_path, source=packed, expected_source_sha256=source_sha256, allow_pending_contacts=allow_pending_contacts)
    report["output_path"] = str(output_path)
    report["output_bytes"] = output_path.stat().st_size
    return report


def write_parameter_authoritative_packed_h5(packed: Mapping[str, Any], output_path: str | Path, *, mhr_layer: Any | None = None, decoder_identity: Mapping[str, str] | None = None, source_path: str | Path | None = None, source_sha256: str | None = None, source_bytes: int | None = None, source_mtime_ns: int | None = None, tolerance_m: float = 2e-6, max_batch_size: int = 256, allow_pending_contacts: bool = False) -> dict[str, Any]:
    if (mhr_layer is None) == (decoder_identity is None):
        raise ValueError("Exactly one of mhr_layer or decoder_identity is required")
    certified = certify_parameter_authoritative_packed(packed, mhr_layer, tolerance_m=tolerance_m, max_batch_size=max_batch_size) if mhr_layer is not None else stamp_parameter_authoritative_packed(packed, decoder_identity)
    return write_packed_h5(certified, output_path, source_path=source_path, source_sha256=source_sha256, source_bytes=source_bytes, source_mtime_ns=source_mtime_ns, allow_pending_contacts=allow_pending_contacts)


def validate_packed_h5(path: str | Path, *, source: Mapping[str, Any] | None = None, expected_source_sha256: str | None = None, allow_pending_contacts: bool = False) -> dict[str, Any]:
    collected = collect_packed_datasets(source, allow_pending_contacts=allow_pending_contacts) if source is not None else None
    with h5py.File(path, "r") as handle:
        return _validate_handle(handle, collected, expected_source_sha256, require_complete=True, allow_pending_contacts=allow_pending_contacts)


def inspect_packed_h5(path: str | Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        report = _validate_handle(handle, None, None, require_complete=True)
        report.update({"path": str(Path(path)), "source_path": str(handle.attrs.get("source_path", "")), "source_bytes": int(handle.attrs.get("source_bytes", -1)), "source_mtime_ns": int(handle.attrs.get("source_mtime_ns", -1)), "output_bytes": Path(path).stat().st_size})
        return report


def sha256_file(path: str | Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def convert_packed_pickle(source_path: str | Path, output_path: str | Path, *, mhr_layer: Any, resume: bool = True, replace: bool = False, tolerance_m: float = MHR_GEOMETRY_CERTIFICATE_TOLERANCE_M, max_batch_size: int = 256) -> dict[str, Any]:
    source_path = Path(source_path)
    output_path = Path(output_path)
    before = source_path.stat()
    source_sha256 = sha256_file(source_path)
    after_hash = source_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after_hash.st_size, after_hash.st_mtime_ns):
        raise RuntimeError(f"Source changed while hashing: {source_path}")
    if output_path.exists():
        if resume:
            existing = inspect_packed_h5(output_path)
            if existing["source_sha256"] != source_sha256:
                raise ValueError(f"Existing output source SHA256 {existing['source_sha256']} does not match current source SHA256 {source_sha256}: {output_path}")
            with source_path.open("rb") as handle:
                packed = pickle.load(handle)
            after_load = source_path.stat()
            if (before.st_size, before.st_mtime_ns) != (after_load.st_size, after_load.st_mtime_ns):
                raise RuntimeError(f"Source changed while loading: {source_path}")
            certified = certify_parameter_authoritative_packed(packed, mhr_layer, tolerance_m=tolerance_m, max_batch_size=max_batch_size)
            validate_packed_h5(output_path, source=certified, expected_source_sha256=source_sha256)
            return {**existing, "action": "skipped", "source_sha256": source_sha256}
        if not replace:
            raise FileExistsError(f"Output already exists: {output_path}")
    with source_path.open("rb") as handle:
        packed = pickle.load(handle)
    after_load = source_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after_load.st_size, after_load.st_mtime_ns):
        raise RuntimeError(f"Source changed while loading: {source_path}")
    report = write_parameter_authoritative_packed_h5(packed, output_path, mhr_layer=mhr_layer, source_path=source_path, source_sha256=source_sha256, source_bytes=before.st_size, source_mtime_ns=before.st_mtime_ns, tolerance_m=tolerance_m, max_batch_size=max_batch_size)
    return {**report, "action": "converted", "source_sha256": source_sha256}
