"""Adapt a Video2Data reconstruction bundle to the inpainting contracts.

The reconstruction bundle is deliberately independent of the learned hand
model: both WiLoR and HaMeR write the same ``result/result.npz`` schema.  This
adapter starts at that stable boundary, runs licensed MANO forward kinematics,
and retargets the resulting joints with the existing Robotic Grounding Sharpa
kinematics.

Execution is opt-in.  With no mode flag (or with ``--preflight``), the CLI only
validates inputs and reports blockers.  ``--execute`` is required to run MANO
and Sharpa and write artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from inpainting.contracts import (
    ROBOT_TRAJECTORY_SCHEMA,
    TRACKING_SCHEMA,
    VideoGeometry,
    validate_robot_trajectory_arrays,
    validate_tracking_arrays,
)
from inpainting.taco_camera import load_taco_camera
from inpainting.video_io import probe_video


ADAPTER_SCHEMA = "v2d.inpainting.video2data-adapter/v1"
DEFAULT_MANO_MODEL_DIR = Path("/home/mverghese/visual_inpainting/mano_v1_2")
DEFAULT_ROBOT_ASSETS_DIR = (
    Path(__file__).resolve().parents[2]
    / "robotic_grounding/source/robotic_grounding/robotic_grounding/assets"
)
RESULT_NPZ = "result.npz"
RESULT_MANIFEST = "manifest.json"
SUPPORTED_HAND_SOURCES = frozenset({"wilor", "hamer"})
_MANO_TRANSFORMS_TO_JOINTS = (
    0,
    13,
    14,
    15,
    15,
    1,
    2,
    3,
    3,
    4,
    5,
    6,
    6,
    10,
    11,
    12,
    12,
    7,
    8,
    9,
    9,
)

_COMMON_KEYS = {
    "camera_to_world_transform": ("float", (None, 4, 4)),
    "camera_is_valid": ("bool", (None,)),
}
_HAND_KEYS = {
    "betas": ("float", (10,)),
    "wrist_orient_in_camera": ("float", (None, 3)),
    "wrist_trans_in_camera": ("float", (None, 3)),
    "finger_pose": ("float", (None, 15, 3)),
    "scale": ("float", (None,)),
    "is_valid": ("bool", (None,)),
}


class AdapterError(RuntimeError):
    """Raised when a bundle cannot be converted safely."""


class UpstreamOutputError(AdapterError):
    """Raised when learned reconstruction output is absent or unusable."""


class DependencyError(AdapterError):
    """Raised when the MANO/Sharpa execution environment is incomplete."""


@dataclass(frozen=True)
class ResultBundle:
    result_dir: Path
    npz_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    frame_count: int
    hand_pose_source: str
    input_mode: str = "result_bundle"
    input_files: tuple[Path, ...] = ()


class ManoBackend(Protocol):
    identity: str

    def forward(
        self,
        *,
        side: str,
        betas: np.ndarray,
        global_orient: np.ndarray,
        finger_pose: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return zero-translation ``joints``, ``joints_wxyz``, and ``vertices``."""


class SharpaBackend(Protocol):
    identity: str

    def retarget(
        self,
        *,
        joints: dict[str, np.ndarray],
        joints_wxyz: dict[str, np.ndarray],
        valid: dict[str, np.ndarray],
        mano_to_robot_scale: float,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return per-side wrist/finger targets and joint names."""


def _is_float(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.floating)


def _validate_typed_shape(
    arrays: dict[str, np.ndarray],
    key: str,
    kind: str,
    shape: tuple[int | None, ...],
    frame_count: int,
) -> np.ndarray:
    if key not in arrays:
        raise UpstreamOutputError(f"{RESULT_NPZ} is missing required array {key!r}")
    array = np.asarray(arrays[key])
    expected = tuple(frame_count if dim is None else dim for dim in shape)
    if array.shape != expected:
        raise UpstreamOutputError(
            f"{key} must have shape {expected}, got {array.shape}"
        )
    if kind == "float" and not _is_float(array):
        raise UpstreamOutputError(f"{key} must have floating dtype, got {array.dtype}")
    if kind == "bool" and array.dtype != np.dtype(np.bool_):
        raise UpstreamOutputError(f"{key} must have boolean dtype, got {array.dtype}")
    return array


def _infer_frame_count(arrays: dict[str, np.ndarray]) -> int:
    key = "camera_to_world_transform"
    if key not in arrays:
        raise UpstreamOutputError(f"{RESULT_NPZ} is missing required array {key!r}")
    array = np.asarray(arrays[key])
    if array.ndim != 3:
        raise UpstreamOutputError(f"{key} must have shape (N,4,4), got {array.shape}")
    if array.shape[0] <= 0:
        raise UpstreamOutputError("Reconstruction bundle contains zero frames")
    return int(array.shape[0])


def _validate_rotation_transforms(transforms: np.ndarray, valid: np.ndarray) -> None:
    if not valid.any():
        return
    selected = np.asarray(transforms[valid], dtype=np.float64)
    if not np.isfinite(selected).all():
        raise UpstreamOutputError(
            "camera_to_world_transform has non-finite values in camera-valid rows"
        )
    expected_bottom = np.broadcast_to(
        np.array([0.0, 0.0, 0.0, 1.0]), selected[:, 3, :].shape
    )
    if not np.allclose(selected[:, 3, :], expected_bottom, atol=1e-5):
        raise UpstreamOutputError("camera_to_world_transform is not homogeneous SE(3)")
    rotations = selected[:, :3, :3]
    identity = np.einsum("nij,nkj->nik", rotations, rotations)
    if not np.allclose(identity, np.eye(3), atol=2e-3):
        raise UpstreamOutputError(
            "camera_to_world_transform rotations are not orthonormal"
        )
    determinant = np.linalg.det(rotations)
    if not np.allclose(determinant, 1.0, atol=2e-3):
        raise UpstreamOutputError(
            "camera_to_world_transform contains a reflection or scaled rotation"
        )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise UpstreamOutputError(f"Malformed JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise UpstreamOutputError(f"Expected a JSON object in {path}")
    return value


def load_result_bundle(
    result_dir: str | Path,
    *,
    expected_frames: int | None = None,
    allow_static_camera: bool = False,
) -> ResultBundle:
    """Load and strictly validate the portable Video2Data result bundle."""

    result_dir = Path(result_dir).expanduser().resolve()
    npz_path = result_dir / RESULT_NPZ
    manifest_path = result_dir / RESULT_MANIFEST
    if not npz_path.is_file() or not manifest_path.is_file():
        missing = [str(p) for p in (npz_path, manifest_path) if not p.is_file()]
        raise UpstreamOutputError(
            "Missing Video2Data reconstruction output: "
            + ", ".join(missing)
            + ". Run the WiLoR/HaMeR reconstruction through its result packaging stage."
        )

    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise UpstreamOutputError(
            f"Unsupported result manifest schema {manifest.get('schema_version')!r}; expected 1"
        )
    if manifest.get("result_npz") != RESULT_NPZ:
        raise UpstreamOutputError(
            f"Manifest result_npz must be {RESULT_NPZ!r}, got {manifest.get('result_npz')!r}"
        )
    if manifest.get("camera_pose_convention") != "camera_to_world":
        raise UpstreamOutputError(
            "Manifest camera_pose_convention must be 'camera_to_world'"
        )
    sources = manifest.get("sources")
    sources = sources if isinstance(sources, dict) else {}
    camera_pose_source = sources.get("camera_to_world_dir")
    explicitly_static = sources.get("camera_is_static") is True
    if not camera_pose_source and not explicitly_static and not allow_static_camera:
        raise UpstreamOutputError(
            "Packaged result has no camera_to_world_dir. The upstream packager "
            "writes identity poses in that fallback, which cannot be labeled as a "
            "world-frame egocentric trajectory. Supply real camera poses, declare "
            "sources.camera_is_static=true, or explicitly pass allow_static_camera."
        )

    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise UpstreamOutputError(f"Cannot read {npz_path}: {exc}") from exc
    frame_count = _infer_frame_count(arrays)
    if manifest.get("n_frames") != frame_count:
        raise UpstreamOutputError(
            f"Manifest n_frames={manifest.get('n_frames')!r} but {RESULT_NPZ} has "
            f"{frame_count} frames"
        )
    if expected_frames is not None and frame_count != expected_frames:
        raise UpstreamOutputError(
            f"Reconstruction has {frame_count} frames but source video has "
            f"{expected_frames}; implicit padding/truncation is forbidden"
        )

    for key, (kind, shape) in _COMMON_KEYS.items():
        _validate_typed_shape(arrays, key, kind, shape, frame_count)
    for side in ("left", "right"):
        for suffix, (kind, shape) in _HAND_KEYS.items():
            _validate_typed_shape(
                arrays, f"hand_{side}_{suffix}", kind, shape, frame_count
            )

    camera_valid = arrays["camera_is_valid"]
    _validate_rotation_transforms(arrays["camera_to_world_transform"], camera_valid)
    for side in ("left", "right"):
        marked_valid = arrays[f"hand_{side}_is_valid"]
        for suffix in (
            "wrist_orient_in_camera",
            "wrist_trans_in_camera",
            "finger_pose",
            "scale",
        ):
            values = arrays[f"hand_{side}_{suffix}"]
            if marked_valid.any() and not np.isfinite(values[marked_valid]).all():
                raise UpstreamOutputError(
                    f"hand_{side}_{suffix} has non-finite values in hand-valid rows"
                )
        scale = arrays[f"hand_{side}_scale"]
        if marked_valid.any() and np.any(scale[marked_valid] <= 0.0):
            raise UpstreamOutputError(
                f"hand_{side}_scale must be positive in valid rows"
            )
        betas = arrays[f"hand_{side}_betas"]
        if marked_valid.any() and not np.isfinite(betas).all():
            raise UpstreamOutputError(f"hand_{side}_betas must be finite")

    raw_valid_count = sum(
        int(np.count_nonzero(arrays[f"hand_{side}_is_valid"]))
        for side in ("left", "right")
    )
    if raw_valid_count == 0:
        raise UpstreamOutputError(
            "The result bundle contains no valid WiLoR/HaMeR hand rows. "
            "The bundle may have been packaged without learned hand outputs."
        )

    hand_pose_source = str(sources.get("hand_pose_source") or "legacy_unspecified")
    if (
        hand_pose_source != "legacy_unspecified"
        and hand_pose_source not in SUPPORTED_HAND_SOURCES
    ):
        raise UpstreamOutputError(
            f"Unsupported hand_pose_source {hand_pose_source!r}; expected WiLoR or HaMeR"
        )
    manifest_keys = manifest.get("keys")
    if isinstance(manifest_keys, list):
        absent_from_manifest = sorted(set(_required_result_keys()) - set(manifest_keys))
        if absent_from_manifest:
            raise UpstreamOutputError(
                f"Manifest keys omits required arrays: {absent_from_manifest}"
            )

    return ResultBundle(
        result_dir=result_dir,
        npz_path=npz_path,
        manifest_path=manifest_path,
        manifest=manifest,
        arrays=arrays,
        frame_count=frame_count,
        hand_pose_source=hand_pose_source,
        input_files=(npz_path, manifest_path),
    )


def _raw_record_array(
    record: dict[str, Any], path: Path, dotted_key: str, shape: tuple[int, ...]
) -> np.ndarray:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise UpstreamOutputError(f"{path}: detection is missing {dotted_key}")
        value = value[part]
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise UpstreamOutputError(f"{path}: {dotted_key} is not numeric") from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise UpstreamOutputError(
            f"{path}: {dotted_key} must be finite shape {shape}, got {array.shape}"
        )
    return array


def _validate_raw_wilor_record(
    record: Any, path: Path, width: int, height: int
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise UpstreamOutputError(f"{path}: every WiLoR detection must be an object")
    if type(record.get("is_right")) is not bool:
        raise UpstreamOutputError(f"{path}: detection is_right must be a boolean")
    image_size = _raw_record_array(record, path, "image_size", (2,))
    if not np.array_equal(image_size, [width, height]):
        raise UpstreamOutputError(
            f"{path}: detection image_size {image_size.tolist()} does not match "
            f"source video {[width, height]}"
        )
    score = record.get("score")
    if not isinstance(score, (int, float)) or not np.isfinite(score):
        raise UpstreamOutputError(f"{path}: detection score must be finite")
    bbox = record.get("bbox")
    if not isinstance(bbox, dict):
        raise UpstreamOutputError(f"{path}: detection bbox must be an object")
    try:
        bounds = np.asarray(
            [bbox[name] for name in ("x0", "y0", "x1", "y1")], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise UpstreamOutputError(
            f"{path}: bbox must contain numeric x0,y0,x1,y1"
        ) from exc
    if (
        not np.isfinite(bounds).all()
        or bounds[2] <= bounds[0]
        or bounds[3] <= bounds[1]
    ):
        raise UpstreamOutputError(
            f"{path}: bbox is non-finite or empty: {bounds.tolist()}"
        )
    betas = _raw_record_array(record, path, "mano.betas", (10,))
    orient = _raw_record_array(record, path, "mano.global_orient", (3,))
    fingers = _raw_record_array(record, path, "mano.hand_pose", (45,)).reshape(15, 3)
    camera_translation = _raw_record_array(record, path, "camera.pred_cam_t_full", (3,))
    focal_value: Any = record
    for part in ("camera", "scaled_focal_length"):
        if not isinstance(focal_value, dict) or part not in focal_value:
            raise UpstreamOutputError(
                f"{path}: detection is missing camera.scaled_focal_length"
            )
        focal_value = focal_value[part]
    if (
        not isinstance(focal_value, (int, float))
        or not np.isfinite(focal_value)
        or focal_value <= 0
    ):
        raise UpstreamOutputError(
            f"{path}: camera.scaled_focal_length must be finite and positive"
        )
    if camera_translation[2] <= 0.0:
        raise UpstreamOutputError(f"{path}: pred_cam_t_full z must be positive")
    return {
        "is_right": record["is_right"],
        "score": float(score),
        "bbox": bounds,
        "betas": betas,
        "global_orient": orient,
        "finger_pose": fingers,
        "pred_cam_t_full": camera_translation,
        "scaled_focal_length": float(focal_value),
    }


def _camera_translation_for_intrinsics(
    pred_cam_t_full: np.ndarray,
    scaled_focal_length: float,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Preserve WiLoR's centroid pixel while moving to the real TACO pinhole."""

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    x_virtual, y_virtual, z_virtual = map(float, pred_cam_t_full)
    u = scaled_focal_length * x_virtual / z_virtual + width / 2.0
    v = scaled_focal_length * y_virtual / z_virtual + height / 2.0
    z = z_virtual * fx / scaled_focal_length
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float32)


def load_wilor_json_bundle(
    wilor_json_dir: str | Path,
    *,
    geometry: VideoGeometry,
    taco_intrinsic: str | Path,
    taco_extrinsic: str | Path,
) -> ResultBundle:
    """Load raw ``v2d_wilor.video_to_hands`` JSON with official TACO camera poses.

    Detection list order is never used as identity.  ``is_right`` is the side
    key, and a frame with multiple detections for one side is rejected as
    ambiguous rather than silently switching tracks.
    """

    json_dir = Path(wilor_json_dir).expanduser().resolve()
    intrinsic_path = Path(taco_intrinsic).expanduser().resolve()
    extrinsic_path = Path(taco_extrinsic).expanduser().resolve()
    if not json_dir.is_dir():
        raise UpstreamOutputError(f"Missing raw WiLoR JSON directory: {json_dir}")
    try:
        camera = load_taco_camera(
            intrinsic_path,
            extrinsic_path,
            expected_frames=geometry.frame_count,
            width=geometry.width,
            height=geometry.height,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise UpstreamOutputError(f"Invalid official TACO camera input: {exc}") from exc

    expected_paths = [
        json_dir / f"{index:06d}.json" for index in range(geometry.frame_count)
    ]
    missing = [path for path in expected_paths if not path.is_file()]
    numeric_paths = sorted(
        path for path in json_dir.glob("*.json") if path.stem.isdigit()
    )
    unexpected = [
        path
        for path in numeric_paths
        if not (0 <= int(path.stem) < geometry.frame_count)
        or path.name != f"{int(path.stem):06d}.json"
    ]
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing {len(missing)} frame files (first: {missing[0]})")
        if unexpected:
            detail.append(
                f"unexpected/noncanonical frame files (first: {unexpected[0]})"
            )
        raise UpstreamOutputError(
            "Raw WiLoR output is not exactly frame-aligned: " + "; ".join(detail)
        )

    n_frames = geometry.frame_count
    world_to_camera = camera.world_to_camera
    camera_to_world = np.linalg.inv(world_to_camera).astype(np.float32)
    arrays: dict[str, np.ndarray] = {
        "camera_to_world_transform": camera_to_world,
        "camera_is_valid": np.ones(n_frames, dtype=np.bool_),
    }
    observations: dict[str, list[tuple[int, dict[str, Any]]]] = {
        "left": [],
        "right": [],
    }
    for frame_index, path in enumerate(expected_paths):
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise UpstreamOutputError(f"Malformed JSON in {path}: {exc}") from exc
        if not isinstance(raw, list):
            raise UpstreamOutputError(f"{path}: expected a JSON detection list")
        by_side: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
        for item in raw:
            record = _validate_raw_wilor_record(
                item, path, geometry.width, geometry.height
            )
            side = "right" if record["is_right"] else "left"
            by_side[side].append(record)
        for side in ("left", "right"):
            if len(by_side[side]) > 1:
                raise UpstreamOutputError(
                    f"{path}: {len(by_side[side])} detections claim side={side}; "
                    "stable identity is ambiguous. Supply external bboxes or clean the detections."
                )
            if by_side[side]:
                observations[side].append((frame_index, by_side[side][0]))

    if not any(observations.values()):
        raise UpstreamOutputError(
            "Raw WiLoR output contains no valid hand detections in any video frame"
        )
    for side in ("left", "right"):
        valid = np.zeros(n_frames, dtype=np.bool_)
        orient = np.zeros((n_frames, 3), dtype=np.float32)
        translation = np.zeros((n_frames, 3), dtype=np.float32)
        fingers = np.zeros((n_frames, 15, 3), dtype=np.float32)
        scale = np.ones(n_frames, dtype=np.float32)
        beta_samples: list[np.ndarray] = []
        for frame_index, record in observations[side]:
            valid[frame_index] = True
            orient[frame_index] = record["global_orient"]
            fingers[frame_index] = record["finger_pose"]
            translation[frame_index] = _camera_translation_for_intrinsics(
                record["pred_cam_t_full"],
                record["scaled_focal_length"],
                camera.intrinsic,
                geometry.width,
                geometry.height,
            )
            beta_samples.append(record["betas"])
        betas = (
            np.median(np.stack(beta_samples), axis=0).astype(np.float32)
            if beta_samples
            else np.zeros(10, dtype=np.float32)
        )
        arrays[f"hand_{side}_betas"] = betas
        arrays[f"hand_{side}_wrist_orient_in_camera"] = orient
        arrays[f"hand_{side}_wrist_trans_in_camera"] = translation
        arrays[f"hand_{side}_finger_pose"] = fingers
        arrays[f"hand_{side}_scale"] = scale
        arrays[f"hand_{side}_is_valid"] = valid

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "n_frames": n_frames,
        "result_npz": "generated_in_memory_from_raw_wilor_json",
        "camera_pose_convention": "camera_to_world",
        "sources": {
            "pipeline": "v2d_wilor.lib.video_to_hands",
            "hand_pose_source": "wilor",
            "adapter_input_mode": "raw_wilor_taco_camera",
            "wilor_json_dir": str(json_dir),
            "taco_intrinsic": str(intrinsic_path),
            "taco_extrinsic": str(extrinsic_path),
            "shape_policy": "per-side median betas over valid detections",
            "hand_scale_policy": "1.0 (raw WiLoR has no depth-alignment hand_scale)",
        },
        "keys": sorted(arrays),
    }
    # Apply the same strict array/SE(3)/valid-row gates as packaged bundles.
    for key, (kind, shape) in _COMMON_KEYS.items():
        _validate_typed_shape(arrays, key, kind, shape, n_frames)
    for side in ("left", "right"):
        for suffix, (kind, shape) in _HAND_KEYS.items():
            _validate_typed_shape(
                arrays, f"hand_{side}_{suffix}", kind, shape, n_frames
            )
    _validate_rotation_transforms(camera_to_world, arrays["camera_is_valid"])
    return ResultBundle(
        result_dir=json_dir,
        npz_path=json_dir,
        manifest_path=intrinsic_path,
        manifest=manifest,
        arrays=arrays,
        frame_count=n_frames,
        hand_pose_source="wilor",
        input_mode="raw_wilor_taco_camera",
        input_files=(json_dir, intrinsic_path, extrinsic_path),
    )


def _required_result_keys() -> list[str]:
    keys = list(_COMMON_KEYS)
    keys.extend(
        f"hand_{side}_{suffix}" for side in ("left", "right") for suffix in _HAND_KEYS
    )
    return keys


def mirror_right_space_params_for_native_left(
    global_orient: np.ndarray, finger_pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert stored right-MANO-space left parameters to native-left space."""

    orient = np.asarray(global_orient).copy()
    fingers = np.asarray(finger_pose).copy()
    orient[..., 1:] *= -1.0
    fingers[..., 1:] *= -1.0
    return orient, fingers


def _rotation_matrices_to_wxyz(matrices: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to normalized scalar-first quaternions."""

    matrices = np.asarray(matrices, dtype=np.float64)
    flat = matrices.reshape(-1, 3, 3)
    quaternions = np.empty((flat.shape[0], 4), dtype=np.float64)
    for index, matrix in enumerate(flat):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            q = np.array(
                [
                    0.25 * s,
                    (matrix[2, 1] - matrix[1, 2]) / s,
                    (matrix[0, 2] - matrix[2, 0]) / s,
                    (matrix[1, 0] - matrix[0, 1]) / s,
                ]
            )
        else:
            diagonal = np.diag(matrix)
            axis = int(np.argmax(diagonal))
            if axis == 0:
                s = (
                    np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0))
                    * 2.0
                )
                q = np.array(
                    [
                        (matrix[2, 1] - matrix[1, 2]) / s,
                        0.25 * s,
                        (matrix[0, 1] + matrix[1, 0]) / s,
                        (matrix[0, 2] + matrix[2, 0]) / s,
                    ]
                )
            elif axis == 1:
                s = (
                    np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0))
                    * 2.0
                )
                q = np.array(
                    [
                        (matrix[0, 2] - matrix[2, 0]) / s,
                        (matrix[0, 1] + matrix[1, 0]) / s,
                        0.25 * s,
                        (matrix[1, 2] + matrix[2, 1]) / s,
                    ]
                )
            else:
                s = (
                    np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0))
                    * 2.0
                )
                q = np.array(
                    [
                        (matrix[1, 0] - matrix[0, 1]) / s,
                        (matrix[0, 2] + matrix[2, 0]) / s,
                        (matrix[1, 2] + matrix[2, 1]) / s,
                        0.25 * s,
                    ]
                )
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm < 1e-12:
            raise AdapterError(
                "Cannot convert a degenerate joint rotation to quaternion"
            )
        q /= norm
        if q[0] < 0.0:
            q = -q
        quaternions[index] = q
    return quaternions.reshape(matrices.shape[:-2] + (4,)).astype(np.float32)


def _wxyz_to_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-12):
        raise AdapterError("MANO backend returned a degenerate joint quaternion")
    q = quaternions / norms
    w, x, y, z = np.moveaxis(q, -1, 0)
    matrices = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    matrices[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[..., 0, 1] = 2.0 * (x * y - w * z)
    matrices[..., 0, 2] = 2.0 * (x * z + w * y)
    matrices[..., 1, 0] = 2.0 * (x * y + w * z)
    matrices[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[..., 1, 2] = 2.0 * (y * z - w * x)
    matrices[..., 2, 0] = 2.0 * (x * z - w * y)
    matrices[..., 2, 1] = 2.0 * (y * z + w * x)
    matrices[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def _backend_array(
    result: dict[str, np.ndarray], key: str, shape: tuple[int, ...], side: str
) -> np.ndarray:
    if key not in result:
        raise AdapterError(f"MANO backend omitted {key!r} for {side}")
    value = np.asarray(result[key])
    if value.shape != shape or not _is_float(value):
        raise AdapterError(
            f"MANO backend {side} {key} must be floating {shape}, got "
            f"{value.shape} {value.dtype}"
        )
    if not np.isfinite(value).all():
        raise AdapterError(f"MANO backend returned non-finite {side} {key}")
    return value


def _mano_world_tracks(
    bundle: ResultBundle, mano_backend: ManoBackend
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    arrays = bundle.arrays
    n_frames = bundle.frame_count
    camera_valid = arrays["camera_is_valid"]
    camera_to_world = np.asarray(arrays["camera_to_world_transform"], dtype=np.float64)
    joints_world: dict[str, np.ndarray] = {}
    quaternions_world: dict[str, np.ndarray] = {}
    effective_valid: dict[str, np.ndarray] = {}

    for side in ("left", "right"):
        valid = np.asarray(
            arrays[f"hand_{side}_is_valid"] & camera_valid, dtype=np.bool_
        )
        effective_valid[side] = valid
        world_joints = np.full((n_frames, 21, 3), np.nan, dtype=np.float32)
        world_quaternions = np.full((n_frames, 21, 4), np.nan, dtype=np.float32)
        indices = np.flatnonzero(valid)
        if indices.size:
            orient = np.asarray(
                arrays[f"hand_{side}_wrist_orient_in_camera"][indices],
                dtype=np.float32,
            )
            fingers = np.asarray(
                arrays[f"hand_{side}_finger_pose"][indices], dtype=np.float32
            )
            if side == "left":
                orient, fingers = mirror_right_space_params_for_native_left(
                    orient, fingers
                )
            result = mano_backend.forward(
                side=side,
                betas=np.asarray(arrays[f"hand_{side}_betas"], dtype=np.float32),
                global_orient=orient,
                finger_pose=fingers,
            )
            count = int(indices.size)
            local_joints = _backend_array(result, "joints", (count, 21, 3), side)
            local_quaternions = _backend_array(
                result, "joints_wxyz", (count, 21, 4), side
            )
            vertices = _backend_array(result, "vertices", (count, 778, 3), side)

            # Alignment hand_scale is a silhouette/depth correction around the
            # posed MANO vertex centroid.  It is not object scale and must be
            # applied before cam_t and camera-to-world composition.
            centroid = vertices.mean(axis=1)
            hand_scale = np.asarray(arrays[f"hand_{side}_scale"][indices])
            scaled_joints = (local_joints - centroid[:, None, :]) * hand_scale[
                :, None, None
            ] + centroid[:, None, :]
            camera_joints = (
                scaled_joints
                + np.asarray(arrays[f"hand_{side}_wrist_trans_in_camera"][indices])[
                    :, None, :
                ]
            )
            rotation_c2w = camera_to_world[indices, :3, :3]
            translation_c2w = camera_to_world[indices, :3, 3]
            world_joints[indices] = (
                np.einsum("nij,nkj->nki", rotation_c2w, camera_joints)
                + translation_c2w[:, None, :]
            ).astype(np.float32)

            local_rotations = _wxyz_to_rotation_matrices(local_quaternions)
            world_rotations = np.einsum("nij,nkjl->nkil", rotation_c2w, local_rotations)
            world_quaternions[indices] = _rotation_matrices_to_wxyz(world_rotations)

        joints_world[side] = world_joints
        quaternions_world[side] = world_quaternions

    if not any(valid.any() for valid in effective_valid.values()):
        raise UpstreamOutputError(
            "No hand row has both valid learned tracking and a valid camera-to-world pose"
        )
    return joints_world, quaternions_world, effective_valid


def tracking_from_bundle(
    bundle: ResultBundle, *, mano_backend: ManoBackend
) -> dict[str, np.ndarray]:
    """Run MANO FK and create the strict common tracking archive arrays."""

    joints, quaternions, valid = _mano_world_tracks(bundle, mano_backend)
    n_frames = bundle.frame_count
    tracking: dict[str, np.ndarray] = {
        "schema_version": np.asarray(TRACKING_SCHEMA),
        "tracker": np.asarray("v2d"),
        "coordinate_frame": np.asarray("world"),
        "frame_indices": np.arange(n_frames, dtype=np.int32),
    }
    for side in ("left", "right"):
        tracking[f"{side}_valid"] = valid[side].copy()
        tracking[f"{side}_wrist_position"] = joints[side][:, 0].copy()
        tracking[f"{side}_wrist_wxyz"] = quaternions[side][:, 0].copy()
        tracking[f"{side}_joints_3d"] = joints[side].copy()
        # Sharpa consumes all MANO joint orientations.  This optional common
        # field lets MANO FK and retargeting run in their existing separate
        # containers without a private intermediate format.
        tracking[f"{side}_joints_wxyz"] = quaternions[side].copy()
    validate_tracking_arrays(tracking, expected_frames=n_frames)
    return tracking


def trajectory_from_tracking(
    tracking: dict[str, np.ndarray],
    *,
    sharpa_backend: SharpaBackend,
    mano_to_robot_scale: float = 1.2,
) -> dict[str, np.ndarray]:
    """Run existing Sharpa IK from a strict MANO-enriched tracking archive."""

    if not np.isfinite(mano_to_robot_scale) or mano_to_robot_scale <= 0.0:
        raise ValueError("mano_to_robot_scale must be finite and positive")
    n_frames = validate_tracking_arrays(tracking)
    coordinate_frame = str(np.asarray(tracking["coordinate_frame"]).item())
    joints: dict[str, np.ndarray] = {}
    quaternions: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        joint_key = f"{side}_joints_3d"
        quaternion_key = f"{side}_joints_wxyz"
        if joint_key not in tracking or quaternion_key not in tracking:
            raise AdapterError(
                f"Sharpa stage requires {joint_key} and {quaternion_key} from MANO FK"
            )
        joints[side] = np.asarray(tracking[joint_key])
        quaternions[side] = np.asarray(tracking[quaternion_key])
        valid[side] = np.asarray(tracking[f"{side}_valid"])
    retargeted = sharpa_backend.retarget(
        joints=joints,
        joints_wxyz=quaternions,
        valid=valid,
        mano_to_robot_scale=float(mano_to_robot_scale),
    )
    trajectory: dict[str, np.ndarray] = {
        "schema_version": np.asarray(ROBOT_TRAJECTORY_SCHEMA),
        "coordinate_frame": np.asarray(coordinate_frame),
        "robot": np.asarray("dexmate_vega"),
        "gripper": np.asarray("sharpa_wave"),
        "frame_indices": np.arange(n_frames, dtype=np.int32),
    }
    for side in ("left", "right"):
        if side not in retargeted:
            raise AdapterError(f"Sharpa backend omitted {side!r} results")
        result = retargeted[side]
        position = np.asarray(result.get("wrist_position"))
        quaternion = np.asarray(result.get("wrist_wxyz"))
        fingers = np.asarray(result.get("finger_joints"))
        names = np.asarray(result.get("finger_joint_names"))
        backend_valid = np.asarray(result.get("valid", valid[side]))
        if position.shape != (n_frames, 3) or not _is_float(position):
            raise AdapterError(f"Sharpa {side} wrist_position must be floating (N,3)")
        if quaternion.shape != (n_frames, 4) or not _is_float(quaternion):
            raise AdapterError(f"Sharpa {side} wrist_wxyz must be floating (N,4)")
        if fingers.ndim != 2 or fingers.shape[0] != n_frames or not _is_float(fingers):
            raise AdapterError(f"Sharpa {side} finger_joints must be floating (N,J)")
        if names.shape != (fingers.shape[1],) or names.dtype.kind not in {"U", "S"}:
            raise AdapterError(f"Sharpa {side} finger_joint_names must be string (J,)")
        if backend_valid.shape != (n_frames,) or backend_valid.dtype != np.dtype(
            np.bool_
        ):
            raise AdapterError(f"Sharpa {side} valid must be boolean (N,)")
        if np.any(backend_valid & ~valid[side]):
            raise AdapterError(f"Sharpa {side} marked an unobserved tracking row valid")
        # The learned validity mask is authoritative.  A backend is never
        # allowed to fill a missing observation from its temporal state. The
        # solver may additionally reject a finite but poor-residual solution.
        invalid = ~backend_valid
        position = position.astype(np.float32, copy=True)
        quaternion = quaternion.astype(np.float32, copy=True)
        fingers = fingers.astype(np.float32, copy=True)
        position[invalid] = np.nan
        quaternion[invalid] = np.nan
        fingers[invalid] = np.nan
        trajectory[f"{side}_valid"] = backend_valid.copy()
        trajectory[f"{side}_wrist_position"] = position
        trajectory[f"{side}_wrist_wxyz"] = quaternion
        trajectory[f"{side}_finger_joints"] = fingers
        trajectory[f"{side}_finger_joint_names"] = names.astype(str)

    validate_robot_trajectory_arrays(trajectory, expected_frames=n_frames)
    return trajectory


def arrays_from_bundle(
    bundle: ResultBundle,
    *,
    mano_backend: ManoBackend,
    sharpa_backend: SharpaBackend,
    mano_to_robot_scale: float = 1.2,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Convert a validated result bundle using injected MANO and Sharpa backends."""

    tracking = tracking_from_bundle(bundle, mano_backend=mano_backend)
    trajectory = trajectory_from_tracking(
        tracking,
        sharpa_backend=sharpa_backend,
        mano_to_robot_scale=mano_to_robot_scale,
    )
    return tracking, trajectory


class ManoTorchBackend:
    """MANO FK matching the existing task-library wrapper, without RG imports."""

    identity = "manotorch.ManoLayer+AxisLayerFK(flat_hand_mean=true,center_idx=null)"

    def __init__(self, mano_model_dir: Path, device: str = "cpu") -> None:
        used_deprecation_shim = False
        try:
            import torch
            from manotorch.manolayer import ManoLayer

            try:
                from manotorch.axislayer import AxisLayerFK
            except ModuleNotFoundError as exc:
                if exc.name != "deprecation":
                    raise
                # Some pinned manotorch wheels omit the warning-only
                # ``deprecation`` dependency. AxisLayerFK itself does not use
                # it; only a legacy class decorator in the same module does.
                import sys
                import types

                shim = types.ModuleType("deprecation")

                def deprecated(*_args: Any, **_kwargs: Any):
                    return lambda value: value

                shim.deprecated = deprecated
                sys.modules["deprecation"] = shim
                used_deprecation_shim = True
                from manotorch.axislayer import AxisLayerFK
        except ImportError as exc:
            raise DependencyError(
                "MANO execution requires torch and manotorch"
            ) from exc
        self._torch = torch
        self.identity = type(self).identity + (
            ";deprecation-warning-shim=identity" if used_deprecation_shim else ""
        )
        self._device = torch.device(device)
        self._layers = {}
        self._axes = {}
        for side in ("left", "right"):
            self._layers[side] = ManoLayer(
                use_pca=False,
                side=side,
                gender="neutral",
                center_idx=None,
                mano_assets_root=str(mano_model_dir),
                flat_hand_mean=True,
            ).to(self._device)
            self._axes[side] = AxisLayerFK(
                side=side, mano_assets_root=str(mano_model_dir)
            ).to(self._device)

    def forward(
        self,
        *,
        side: str,
        betas: np.ndarray,
        global_orient: np.ndarray,
        finger_pose: np.ndarray,
    ) -> dict[str, np.ndarray]:
        torch = self._torch
        count = int(global_orient.shape[0])
        with torch.no_grad():
            betas_t = torch.as_tensor(betas, dtype=torch.float32, device=self._device)
            if betas_t.ndim == 1:
                betas_t = betas_t.unsqueeze(0).expand(count, -1)
            pose = torch.cat(
                [
                    torch.as_tensor(
                        global_orient, dtype=torch.float32, device=self._device
                    ),
                    torch.as_tensor(
                        finger_pose, dtype=torch.float32, device=self._device
                    ).reshape(count, 45),
                ],
                dim=-1,
            )
            result = self._layers[side](pose_coeffs=pose, betas=betas_t)
            transforms, _, _ = self._axes[side](result.transforms_abs)
            rotations = transforms[:, _MANO_TRANSFORMS_TO_JOINTS, :3, :3]
        return {
            "joints": result.joints.detach().cpu().numpy(),
            "joints_wxyz": _rotation_matrices_to_wxyz(rotations.detach().cpu().numpy()),
            "vertices": result.verts.detach().cpu().numpy(),
        }


class ExistingSharpaBackend:
    """Use Robotic Grounding's existing Sharpa mappings and IK implementation."""

    identity = "robotic_grounding.SharpaHandKinematics.compute"

    def __init__(
        self,
        device: str = "cpu",
        *,
        robot_assets_dir: str | Path | None = None,
        max_frame_task_error_m: float = 0.07,
    ) -> None:
        if not np.isfinite(max_frame_task_error_m) or max_frame_task_error_m <= 0.0:
            raise ValueError("max_frame_task_error_m must be finite and positive")
        try:
            import torch
            from robotic_grounding.retarget.hand_kinematics import (
                SharpaHandKinematics,
            )
            from robotic_grounding.retarget.retarget_utils import (
                setup_sharpa_kinematics,
                wrist_pose_from_mano_joint0,
            )
        except ImportError as exc:
            raise DependencyError(
                "Sharpa retargeting requires the Robotic Grounding environment "
                "(torch, scipy, pinocchio, pink, qpsolvers/daqp, and assets)"
            ) from exc
        self._torch = torch
        self._device = torch.device(device)
        self._wrist_pose = wrist_pose_from_mano_joint0
        self.max_frame_task_error_m = float(max_frame_task_error_m)
        self.robot_assets_dir = (
            Path(robot_assets_dir).expanduser().resolve()
            if robot_assets_dir is not None
            else None
        )
        if self.robot_assets_dir is None:
            self._kinematics = {
                side: setup_sharpa_kinematics(
                    side=side, frame_tasks_converged_threshold=1e-6
                )
                for side in ("left", "right")
            }
        else:
            xml_dir = self.robot_assets_dir / "xmls" / "sharpawave"
            self._kinematics = {
                side: SharpaHandKinematics(
                    side=side,
                    robot_asset_path=str(xml_dir / f"{side}_sharpawave.xml"),
                    source_model="mano",
                    use_relative_frames=False,
                    max_iter=100,
                    frequency=200.0,
                    frame_tasks_converged_threshold=1e-6,
                )
                for side in ("left", "right")
            }
        self.diagnostics: dict[str, dict[str, Any]] = {}

    def _retarget_side(
        self,
        side: str,
        joints: np.ndarray,
        joints_wxyz: np.ndarray,
        valid: np.ndarray,
        scale: float,
    ) -> dict[str, np.ndarray]:
        n_frames = int(valid.shape[0])
        kinematics = self._kinematics[side]
        names = np.asarray(
            list(kinematics.robot_finger_joint_names.values()), dtype=str
        )
        wrist_position = np.full((n_frames, 3), np.nan, dtype=np.float32)
        wrist_wxyz = np.full((n_frames, 4), np.nan, dtype=np.float32)
        finger_joints = np.full((n_frames, len(names)), np.nan, dtype=np.float32)
        solver_valid = np.zeros(n_frames, dtype=np.bool_)
        accepted_residuals: list[float] = []
        rejected_residuals: list[float] = []
        iteration_counts: list[int] = []
        previous_qpos: np.ndarray | None = None
        for frame_index in range(n_frames):
            if not valid[frame_index]:
                # Reset at every gap.  The next observed frame is initialized
                # from that observation, never from an unobserved predecessor.
                previous_qpos = None
                continue
            joints_t = self._torch.as_tensor(
                joints[frame_index], dtype=self._torch.float32, device=self._device
            )
            quaternion_t = self._torch.as_tensor(
                joints_wxyz[frame_index],
                dtype=self._torch.float32,
                device=self._device,
            )
            if previous_qpos is None:
                position, quaternion_xyzw = self._wrist_pose(
                    joints[frame_index, 0], joints_wxyz[frame_index, 0]
                )
                qpos = kinematics.robot.q0.copy()
                qpos[:3] = position
                qpos[3:7] = quaternion_xyzw
            else:
                qpos = previous_qpos.copy()
            result = kinematics.compute(
                joints_t,
                quaternion_t,
                source_to_robot_scale=scale,
                qpos=qpos,
            )
            solved = np.asarray(result["q"], dtype=np.float64)
            if solved.ndim != 1 or solved.shape[0] != 7 + len(names):
                raise AdapterError(
                    f"Sharpa {side} frame {frame_index} returned q shape {solved.shape}"
                )
            if not np.isfinite(solved).all():
                raise AdapterError(
                    f"Sharpa {side} frame {frame_index} returned non-finite q"
                )
            errors = np.asarray(result.get("frame_task_errors"), dtype=np.float64)
            if errors.ndim != 1 or errors.size == 0 or not np.isfinite(errors).all():
                raise AdapterError(
                    f"Sharpa {side} frame {frame_index} returned invalid frame_task_errors"
                )
            max_error = float(np.max(errors))
            iteration_counts.append(int(result.get("num_optimization_iterations", -1)))
            if max_error > self.max_frame_task_error_m:
                # Do not let a finite but poor IK result become an observation
                # or seed the next frame.
                rejected_residuals.append(max_error)
                previous_qpos = None
                continue
            accepted_residuals.append(max_error)
            previous_qpos = solved.copy()
            solver_valid[frame_index] = True
            wrist_position[frame_index] = solved[:3]
            wrist_wxyz[frame_index] = solved[3:7][[3, 0, 1, 2]]
            finger_joints[frame_index] = solved[7:]
        residuals = accepted_residuals + rejected_residuals
        residual_array = np.asarray(residuals, dtype=np.float64)
        self.diagnostics[side] = {
            "input_valid_frames": int(np.count_nonzero(valid)),
            "accepted_frames": int(np.count_nonzero(solver_valid)),
            "rejected_frames": len(rejected_residuals),
            "max_frame_task_error_threshold_m": self.max_frame_task_error_m,
            "max_frame_task_error_m": max(residuals) if residuals else None,
            "median_frame_task_error_m": (
                float(np.median(residual_array)) if residuals else None
            ),
            "p95_frame_task_error_m": (
                float(np.percentile(residual_array, 95.0)) if residuals else None
            ),
            "mean_accepted_max_frame_task_error_m": (
                float(np.mean(accepted_residuals)) if accepted_residuals else None
            ),
            "max_rejected_frame_task_error_m": (
                max(rejected_residuals) if rejected_residuals else None
            ),
            "max_optimization_iterations": (
                max(iteration_counts) if iteration_counts else None
            ),
        }
        return {
            "valid": solver_valid,
            "wrist_position": wrist_position,
            "wrist_wxyz": wrist_wxyz,
            "finger_joints": finger_joints,
            "finger_joint_names": names,
        }

    def retarget(
        self,
        *,
        joints: dict[str, np.ndarray],
        joints_wxyz: dict[str, np.ndarray],
        valid: dict[str, np.ndarray],
        mano_to_robot_scale: float,
    ) -> dict[str, dict[str, np.ndarray]]:
        return {
            side: self._retarget_side(
                side,
                joints[side],
                joints_wxyz[side],
                valid[side],
                mano_to_robot_scale,
            )
            for side in ("left", "right")
        }


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(64).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            )
    except OSError:
        return False


def sharpa_asset_blockers(robot_assets_dir: str | Path) -> list[dict[str, str]]:
    """Validate exactly the XML and mesh files consumed by Sharpa IK."""

    robot_assets_dir = Path(robot_assets_dir).expanduser().resolve()
    blockers: list[dict[str, str]] = []
    xml_dir = robot_assets_dir / "xmls" / "sharpawave"
    for side in ("left", "right"):
        xml_path = xml_dir / f"{side}_sharpawave.xml"
        if not xml_path.is_file():
            blockers.append(
                {
                    "code": "missing_sharpa_xml",
                    "path": str(xml_path),
                    "detail": "Existing Sharpa kinematics requires this MuJoCo XML.",
                }
            )
            continue
        try:
            root = ET.parse(xml_path).getroot()
        except (ET.ParseError, OSError) as exc:
            blockers.append(
                {
                    "code": "invalid_sharpa_xml",
                    "path": str(xml_path),
                    "detail": str(exc),
                }
            )
            continue
        compiler = root.find("compiler")
        meshdir_value = compiler.get("meshdir") if compiler is not None else None
        if not meshdir_value:
            blockers.append(
                {
                    "code": "missing_sharpa_meshdir",
                    "path": str(xml_path),
                    "detail": "MuJoCo XML compiler must declare meshdir.",
                }
            )
            continue
        mesh_dir = (xml_path.parent / meshdir_value).resolve()
        for mesh in root.findall("./asset/mesh"):
            filename = mesh.get("file")
            if not filename:
                continue
            mesh_path = mesh_dir / filename
            if not mesh_path.is_file():
                blockers.append(
                    {
                        "code": "missing_sharpa_mesh",
                        "path": str(mesh_path),
                        "detail": f"Referenced by {xml_path.name}.",
                    }
                )
            elif _is_lfs_pointer(mesh_path):
                blockers.append(
                    {
                        "code": "sharpa_asset_is_lfs_pointer",
                        "path": str(mesh_path),
                        "detail": f"Referenced by {xml_path.name}; hydrate before IK.",
                    }
                )
    return blockers


def sharpa_asset_artifacts(robot_assets_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Fingerprint every XML and mesh file loaded by the Sharpa hand models.

    The keys are paths relative to ``robot_assets_dir`` so the identity is
    stable across read-only container mount points.  Callers must run
    :func:`sharpa_asset_blockers` first; this function intentionally refuses a
    partial inventory rather than producing provenance for invalid assets.
    """

    robot_assets_dir = Path(robot_assets_dir).expanduser().resolve()
    blockers = sharpa_asset_blockers(robot_assets_dir)
    if blockers:
        raise AdapterError(
            "Sharpa asset validation failed: "
            + "; ".join(item["detail"] for item in blockers)
        )

    paths: set[Path] = set()
    xml_dir = robot_assets_dir / "xmls" / "sharpawave"
    for side in ("left", "right"):
        xml_path = (xml_dir / f"{side}_sharpawave.xml").resolve()
        paths.add(xml_path)
        root = ET.parse(xml_path).getroot()
        compiler = root.find("compiler")
        # ``sharpa_asset_blockers`` has already required this attribute.
        assert compiler is not None and compiler.get("meshdir")
        mesh_dir = (xml_path.parent / str(compiler.get("meshdir"))).resolve()
        for mesh in root.findall("./asset/mesh"):
            filename = mesh.get("file")
            if filename:
                paths.add((mesh_dir / filename).resolve())

    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(paths, key=lambda item: str(item)):
        try:
            relative = path.relative_to(robot_assets_dir).as_posix()
        except ValueError as exc:
            raise AdapterError(
                f"Sharpa asset resolves outside robot_assets_dir: {path}"
            ) from exc
        artifacts[relative] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return artifacts


def preflight(
    *,
    result_dir: str | Path | None = None,
    wilor_json_dir: str | Path | None = None,
    taco_intrinsic: str | Path | None = None,
    taco_extrinsic: str | Path | None = None,
    source_video: str | Path,
    output_dir: str | Path,
    mano_model_dir: str | Path = DEFAULT_MANO_MODEL_DIR,
    robot_assets_dir: str | Path = DEFAULT_ROBOT_ASSETS_DIR,
    allow_static_camera: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Return a JSON-serializable, non-mutating readiness report."""

    if (result_dir is None) == (wilor_json_dir is None):
        raise ValueError("Supply exactly one of result_dir or wilor_json_dir")
    result_dir_path = (
        Path(result_dir).expanduser().resolve() if result_dir is not None else None
    )
    wilor_json_path = (
        Path(wilor_json_dir).expanduser().resolve()
        if wilor_json_dir is not None
        else None
    )
    source_video = Path(source_video).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    mano_model_dir = Path(mano_model_dir).expanduser().resolve()
    robot_assets_dir = Path(robot_assets_dir).expanduser().resolve()
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    geometry: VideoGeometry | None = None
    bundle: ResultBundle | None = None

    if not source_video.is_file():
        blockers.append(
            {
                "code": "missing_source_video",
                "path": str(source_video),
                "detail": "FPS and frame alignment must be probed from the original RGB video.",
            }
        )
    else:
        try:
            geometry = probe_video(source_video)
        except Exception as exc:  # ffprobe errors are input blockers, not crashes.
            blockers.append(
                {
                    "code": "source_video_probe_failed",
                    "path": str(source_video),
                    "detail": str(exc),
                }
            )

    if result_dir_path is not None:
        if (
            not (result_dir_path / RESULT_NPZ).is_file()
            or not (result_dir_path / RESULT_MANIFEST).is_file()
        ):
            blockers.append(
                {
                    "code": "missing_reconstruction_bundle",
                    "path": str(result_dir_path),
                    "detail": (
                        f"Expected both {result_dir_path / RESULT_NPZ} and "
                        f"{result_dir_path / RESULT_MANIFEST}; run WiLoR/HaMeR result packaging."
                    ),
                }
            )
        else:
            try:
                bundle = load_result_bundle(
                    result_dir_path,
                    expected_frames=geometry.frame_count
                    if geometry is not None
                    else None,
                    allow_static_camera=allow_static_camera,
                )
            except AdapterError as exc:
                blockers.append(
                    {
                        "code": "invalid_reconstruction_bundle",
                        "path": str(result_dir_path),
                        "detail": str(exc),
                    }
                )
    else:
        assert wilor_json_path is not None
        calibration_paths = {
            "taco_intrinsic": (
                Path(taco_intrinsic).expanduser().resolve()
                if taco_intrinsic is not None
                else None
            ),
            "taco_extrinsic": (
                Path(taco_extrinsic).expanduser().resolve()
                if taco_extrinsic is not None
                else None
            ),
        }
        if not wilor_json_path.is_dir():
            blockers.append(
                {
                    "code": "missing_wilor_json_dir",
                    "path": str(wilor_json_path),
                    "detail": "Expected 000000.json through the final source-video frame.",
                }
            )
        for name, path in calibration_paths.items():
            if path is None or not path.is_file():
                blockers.append(
                    {
                        "code": f"missing_{name}",
                        "path": str(path) if path else "",
                        "detail": "Raw camera-frame WiLoR requires official TACO calibration.",
                    }
                )
        if (
            geometry is not None
            and wilor_json_path.is_dir()
            and all(
                path is not None and path.is_file()
                for path in calibration_paths.values()
            )
        ):
            try:
                bundle = load_wilor_json_bundle(
                    wilor_json_path,
                    geometry=geometry,
                    taco_intrinsic=calibration_paths["taco_intrinsic"],
                    taco_extrinsic=calibration_paths["taco_extrinsic"],
                )
            except AdapterError as exc:
                blockers.append(
                    {
                        "code": "invalid_wilor_json_input",
                        "path": str(wilor_json_path),
                        "detail": str(exc),
                    }
                )

    mano_files = [
        mano_model_dir / "models" / f"MANO_{side}.pkl" for side in ("LEFT", "RIGHT")
    ]
    for path in mano_files:
        if not path.is_file():
            blockers.append(
                {
                    "code": "missing_licensed_mano_model",
                    "path": str(path),
                    "detail": "MANO is separately licensed and is never downloaded by this adapter.",
                }
            )

    required_modules = (
        "torch",
        "manotorch",
        "scipy",
        "pinocchio",
        "pink",
        "loop_rate_limiters",
        "qpsolvers",
        "daqp",
    )
    missing_modules = [name for name in required_modules if not _module_available(name)]
    if missing_modules:
        blockers.append(
            {
                "code": "missing_execution_dependencies",
                "path": "PYTHONPATH/environment",
                "detail": "Missing Python modules: " + ", ".join(missing_modules),
            }
        )
    if not _module_available("robotic_grounding.retarget.retarget_utils"):
        blockers.append(
            {
                "code": "missing_sharpa_retarget_package",
                "path": "PYTHONPATH/environment",
                "detail": "robotic_grounding.retarget.retarget_utils is not importable.",
            }
        )

    blockers.extend(sharpa_asset_blockers(robot_assets_dir))

    outputs = [
        output_dir / "tracking.npz",
        output_dir / "robot_trajectory.npz",
        output_dir / "adapter_metadata.json",
    ]
    existing_outputs = [path for path in outputs if path.exists()]
    if existing_outputs and not overwrite:
        blockers.append(
            {
                "code": "outputs_exist",
                "path": str(output_dir),
                "detail": "Refusing to replace: "
                + ", ".join(path.name for path in existing_outputs),
            }
        )
    if bundle is not None and bundle.hand_pose_source == "legacy_unspecified":
        warnings.append(
            {
                "code": "legacy_hand_source_unspecified",
                "path": str(bundle.manifest_path),
                "detail": "Bundle predates hand_pose_source provenance; common schema is still compatible.",
            }
        )
    effective_valid: dict[str, int] | None = None
    if bundle is not None:
        camera_valid = np.asarray(bundle.arrays["camera_is_valid"])
        effective_valid = {
            side: int(
                np.count_nonzero(
                    np.asarray(bundle.arrays[f"hand_{side}_is_valid"]) & camera_valid
                )
            )
            for side in ("left", "right")
        }
        if sum(effective_valid.values()) == 0:
            blockers.append(
                {
                    "code": "no_effective_hand_observations",
                    "path": str(bundle.result_dir),
                    "detail": "No hand row is valid at a camera-valid frame.",
                }
            )

    report: dict[str, Any] = {
        "schema_version": ADAPTER_SCHEMA,
        "mode": "preflight",
        "state": "ready" if not blockers else "blocked",
        "input_mode": "result_bundle"
        if result_dir_path is not None
        else "raw_wilor_taco_camera",
        "expected_upstream_files": (
            [
                str(result_dir_path / RESULT_NPZ),
                str(result_dir_path / RESULT_MANIFEST),
            ]
            if result_dir_path is not None
            else [
                str(wilor_json_path / "000000.json"),
                str(Path(taco_intrinsic).expanduser().resolve())
                if taco_intrinsic
                else "",
                str(Path(taco_extrinsic).expanduser().resolve())
                if taco_extrinsic
                else "",
            ]
        ),
        "source_video": str(source_video),
        "output_dir": str(output_dir),
        "mano_model_dir": str(mano_model_dir),
        "mano_files": [str(path) for path in mano_files],
        "robot_assets_dir": str(robot_assets_dir),
        "allow_static_camera": bool(allow_static_camera),
        "outputs": [str(path) for path in outputs],
        "blockers": blockers,
        "warnings": warnings,
    }
    if result_dir_path is not None:
        report["result_dir"] = str(result_dir_path)
    else:
        report["wilor_json_dir"] = str(wilor_json_path)
    if geometry is not None:
        report["video"] = geometry.as_dict()
    if bundle is not None:
        report["bundle"] = {
            "frame_count": bundle.frame_count,
            "hand_pose_source": bundle.hand_pose_source,
            "valid_frames": {
                side: int(np.count_nonzero(bundle.arrays[f"hand_{side}_is_valid"]))
                for side in ("left", "right")
            },
            "effective_valid_frames": effective_valid,
        }
    return report


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_sequence_id(result_path: str | Path) -> str:
    """Derive a clip identifier without returning generic stage directories."""

    candidate = Path(result_path).expanduser().resolve()
    generic = {
        "result",
        "results",
        "output",
        "outputs",
        "tracking",
        "wilor_raw",
        "v2d",
        "reconstruction",
    }
    while candidate.name.lower() in generic and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate.name


def convert(
    *,
    result_dir: str | Path | None = None,
    wilor_json_dir: str | Path | None = None,
    taco_intrinsic: str | Path | None = None,
    taco_extrinsic: str | Path | None = None,
    source_video: str | Path,
    output_dir: str | Path,
    mano_model_dir: str | Path = DEFAULT_MANO_MODEL_DIR,
    robot_assets_dir: str | Path = DEFAULT_ROBOT_ASSETS_DIR,
    device: str = "cpu",
    mano_to_robot_scale: float = 1.2,
    max_frame_task_error_m: float = 0.07,
    sequence_id: str | None = None,
    allow_static_camera: bool = False,
    overwrite: bool = False,
    mano_backend: ManoBackend | None = None,
    sharpa_backend: SharpaBackend | None = None,
) -> dict[str, Any]:
    """Execute conversion and write metadata last as the bundle commit marker."""

    source_video = Path(source_video).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    mano_model_dir = Path(mano_model_dir).expanduser().resolve()
    if mano_backend is None or sharpa_backend is None:
        report = preflight(
            result_dir=result_dir,
            wilor_json_dir=wilor_json_dir,
            taco_intrinsic=taco_intrinsic,
            taco_extrinsic=taco_extrinsic,
            source_video=source_video,
            output_dir=output_dir,
            mano_model_dir=mano_model_dir,
            robot_assets_dir=robot_assets_dir,
            allow_static_camera=allow_static_camera,
            overwrite=overwrite,
        )
        if report["state"] != "ready":
            details = "; ".join(item["detail"] for item in report["blockers"])
            raise DependencyError(f"Video2Data adapter preflight blocked: {details}")
    geometry = probe_video(source_video)
    if (result_dir is None) == (wilor_json_dir is None):
        raise ValueError("Supply exactly one of result_dir or wilor_json_dir")
    if result_dir is not None:
        bundle = load_result_bundle(
            result_dir,
            expected_frames=geometry.frame_count,
            allow_static_camera=allow_static_camera,
        )
    else:
        if taco_intrinsic is None or taco_extrinsic is None:
            raise ValueError(
                "Raw WiLoR input requires taco_intrinsic and taco_extrinsic"
            )
        bundle = load_wilor_json_bundle(
            wilor_json_dir,
            geometry=geometry,
            taco_intrinsic=taco_intrinsic,
            taco_extrinsic=taco_extrinsic,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_path = output_dir / "tracking.npz"
    trajectory_path = output_dir / "robot_trajectory.npz"
    metadata_path = output_dir / "adapter_metadata.json"
    existing = [
        path
        for path in (tracking_path, trajectory_path, metadata_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to replace existing outputs: " + ", ".join(map(str, existing))
        )

    if mano_backend is None:
        mano_backend = ManoTorchBackend(mano_model_dir, device=device)
    if sharpa_backend is None:
        sharpa_backend = ExistingSharpaBackend(
            device=device,
            robot_assets_dir=robot_assets_dir,
            max_frame_task_error_m=max_frame_task_error_m,
        )
    tracking, trajectory = arrays_from_bundle(
        bundle,
        mano_backend=mano_backend,
        sharpa_backend=sharpa_backend,
        mano_to_robot_scale=mano_to_robot_scale,
    )
    tracking_temporary = tracking_path.with_name(
        f".{tracking_path.name}.{os.getpid()}.partial.npz"
    )
    trajectory_temporary = trajectory_path.with_name(
        f".{trajectory_path.name}.{os.getpid()}.partial.npz"
    )
    try:
        np.savez_compressed(tracking_temporary, **tracking)
        np.savez_compressed(trajectory_temporary, **trajectory)
        # Validate all staged bytes before invalidating an older commit marker.
        with np.load(tracking_temporary, allow_pickle=False) as archive:
            validate_tracking_arrays(
                dict(archive), expected_frames=geometry.frame_count
            )
        with np.load(trajectory_temporary, allow_pickle=False) as archive:
            validate_robot_trajectory_arrays(
                dict(archive), expected_frames=geometry.frame_count
            )
    except Exception:
        tracking_temporary.unlink(missing_ok=True)
        trajectory_temporary.unlink(missing_ok=True)
        raise

    metadata: dict[str, Any] = {
        "schema_version": ADAPTER_SCHEMA,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "adapter": "video2data_result_bundle",
        "sequence_id": sequence_id or default_sequence_id(bundle.result_dir),
        "tracker": "v2d",
        "coordinate_frame": "world",
        "source_video": str(source_video),
        "video": geometry.as_dict(),
        "input_mode": bundle.input_mode,
        "input_files": [str(path) for path in bundle.input_files],
        "hand_pose_source": bundle.hand_pose_source,
        "mano_model_dir": str(mano_model_dir),
        "mano_backend": mano_backend.identity,
        "sharpa_backend": sharpa_backend.identity,
        "mano_to_robot_scale": float(mano_to_robot_scale),
        "max_frame_task_error_m": float(max_frame_task_error_m),
        "corrections": {
            "validity": "raw hand validity AND camera validity; invalid rows preserved as NaN",
            "hand_scale": "applied around posed MANO vertex centroid before camera translation",
            "fps": "probed from source video; never hardcoded",
            "left_handedness": "right-space axis angles mirrored into native-left MANO convention",
            "frame_alignment": "exact N equality; no padding, interpolation, or truncation",
            "temporal_ik": "reset after every invalid frame",
            "solver_quality": "rows above max_frame_task_error_m rejected and temporal seed reset",
        },
        "valid_frames": {
            side: int(np.count_nonzero(tracking[f"{side}_valid"]))
            for side in ("left", "right")
        },
        "outputs": {
            "tracking": {
                "path": str(tracking_path),
                "size_bytes": tracking_temporary.stat().st_size,
                "sha256": _sha256(tracking_temporary),
            },
            "robot_trajectory": {
                "path": str(trajectory_path),
                "size_bytes": trajectory_temporary.stat().st_size,
                "sha256": _sha256(trajectory_temporary),
            },
        },
    }
    if bundle.input_mode == "result_bundle":
        metadata["result_npz"] = str(bundle.npz_path)
        metadata["result_manifest"] = str(bundle.manifest_path)
    else:
        metadata["wilor_json_dir"] = str(bundle.result_dir)
        metadata["taco_intrinsic"] = str(taco_intrinsic)
        metadata["taco_extrinsic"] = str(taco_extrinsic)
    diagnostics = getattr(sharpa_backend, "diagnostics", None)
    if diagnostics is not None:
        metadata["sharpa_quality"] = diagnostics
    # The JSON is the bundle commit marker. Remove any old marker before the
    # first artifact replacement so a mid-commit failure can never present a
    # stale complete sidecar for mixed generations.
    try:
        if metadata_path.exists():
            metadata_path.unlink()
        tracking_temporary.replace(tracking_path)
        trajectory_temporary.replace(trajectory_path)
        _atomic_json(metadata_path, metadata)
    finally:
        tracking_temporary.unlink(missing_ok=True)
        trajectory_temporary.unlink(missing_ok=True)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--result-dir",
        type=Path,
        help="Packaged WiLoR/HaMeR result/ directory.",
    )
    inputs.add_argument(
        "--wilor-json-dir",
        type=Path,
        help="Raw v2d_wilor.video_to_hands per-frame JSON directory.",
    )
    parser.add_argument(
        "--taco-intrinsic",
        type=Path,
        help="Official egocentric_intrinsic.txt (required with --wilor-json-dir).",
    )
    parser.add_argument(
        "--taco-extrinsic",
        type=Path,
        help="Official egocentric_frame_extrinsic.npy (required with --wilor-json-dir).",
    )
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mano-model-dir", type=Path, default=DEFAULT_MANO_MODEL_DIR)
    parser.add_argument(
        "--robot-assets-dir", type=Path, default=DEFAULT_ROBOT_ASSETS_DIR
    )
    parser.add_argument("--sequence-id")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mano-to-robot-scale", type=float, default=1.2)
    parser.add_argument("--max-frame-task-error-m", type=float, default=0.07)
    parser.add_argument(
        "--allow-static-camera",
        action="store_true",
        help="Authorize an identity-camera packaged bundle for a truly static camera.",
    )
    parser.add_argument("--overwrite", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Validate and print a readiness report without writing (default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Run MANO + Sharpa and write the strict artifacts.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not args.execute:
        report = preflight(
            result_dir=args.result_dir,
            wilor_json_dir=args.wilor_json_dir,
            taco_intrinsic=args.taco_intrinsic,
            taco_extrinsic=args.taco_extrinsic,
            source_video=args.source_video,
            output_dir=args.output_dir,
            mano_model_dir=args.mano_model_dir,
            robot_assets_dir=args.robot_assets_dir,
            allow_static_camera=args.allow_static_camera,
            overwrite=args.overwrite,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["state"] == "ready" else 2)
    try:
        metadata = convert(
            result_dir=args.result_dir,
            wilor_json_dir=args.wilor_json_dir,
            taco_intrinsic=args.taco_intrinsic,
            taco_extrinsic=args.taco_extrinsic,
            source_video=args.source_video,
            output_dir=args.output_dir,
            mano_model_dir=args.mano_model_dir,
            robot_assets_dir=args.robot_assets_dir,
            device=args.device,
            mano_to_robot_scale=args.mano_to_robot_scale,
            max_frame_task_error_m=args.max_frame_task_error_m,
            sequence_id=args.sequence_id,
            allow_static_camera=args.allow_static_camera,
            overwrite=args.overwrite,
        )
    except (AdapterError, FileExistsError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(
        f"Converted {metadata['sequence_id']} ({metadata['video']['frame_count']} frames) "
        f"-> {metadata['outputs']['tracking']['path']}"
    )


if __name__ == "__main__":
    main()
