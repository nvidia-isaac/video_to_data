from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .rotations import rot6d_to_rotmat_np, rotmat_to_6d_np


CAMERA_AXIS_FLIP = np.diag([1.0, -1.0, -1.0])
MHR_ROOT_JOINT_INDEX = 1
MHR_INIT_ROOT_REVISION = "sam3d_camera_axis_world_root_v1"
MHR_INIT_ROOT_CONVENTION = "mhr_decoded_world_from_sam3d_camera_root"
MHR_INIT_ROOT_LEGACY_CONVENTION = "legacy_camera_to_world_times_sam3d_camera_root"
MHR_INIT_TRANSLATION_REVISION = "mhr_root_joint_1_pivot_translation_v1"
MHR_INIT_TRANSLATION_CONVENTION = "decode_consistent_translation_across_rigid_frames"
MHR_INIT_TRANSLATION_LEGACY_CONVENTION = "legacy_translation_transformed_as_point"


def mhr_init_root_metadata() -> dict[str, str]:
    return {"mhr_init_root_revision": MHR_INIT_ROOT_REVISION, "mhr_init_root_convention": MHR_INIT_ROOT_CONVENTION}


def mhr_init_translation_metadata() -> dict[str, str]:
    return {"mhr_init_translation_revision": MHR_INIT_TRANSLATION_REVISION, "mhr_init_translation_convention": MHR_INIT_TRANSLATION_CONVENTION}


def mhr_init_parameter_frame_metadata() -> dict[str, str]:
    return {**mhr_init_root_metadata(), **mhr_init_translation_metadata()}


def stamp_mhr_init_root_metadata(metadata: Mapping[str, Any], **evidence: Any) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"MHR metadata must be a mapping, got {type(metadata).__name__}")
    stamped = dict(metadata)
    stamped.update(mhr_init_root_metadata())
    if evidence:
        stamped["mhr_init_root_evidence"] = dict(evidence)
    if "mhr_init_metadata" in stamped:
        values = stamped["mhr_init_metadata"]
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"mhr_init_metadata must be a list or tuple, got {type(values).__name__}")
        stamped["mhr_init_metadata"] = [stamp_mhr_init_root_metadata(value, **evidence) for value in values]
    return stamped


def stamp_mhr_init_translation_metadata(metadata: Mapping[str, Any], **evidence: Any) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"MHR metadata must be a mapping, got {type(metadata).__name__}")
    stamped = dict(metadata)
    stamped.update(mhr_init_translation_metadata())
    if evidence:
        stamped["mhr_init_translation_evidence"] = dict(evidence)
    if "mhr_init_metadata" in stamped:
        values = stamped["mhr_init_metadata"]
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"mhr_init_metadata must be a list or tuple, got {type(values).__name__}")
        stamped["mhr_init_metadata"] = [stamp_mhr_init_translation_metadata(value, **evidence) for value in values]
    return stamped


def stamp_mhr_init_parameter_frame_metadata(metadata: Mapping[str, Any], **evidence: Any) -> dict[str, Any]:
    return stamp_mhr_init_translation_metadata(stamp_mhr_init_root_metadata(metadata, **evidence), **evidence)


def mhr_init_root_revision_required(metadata: Mapping[str, Any]) -> bool:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"MHR metadata must be a mapping, got {type(metadata).__name__}")
    sources = (metadata.get("source"), metadata.get("mhr_init_source"), metadata.get("init_geometry_source"))
    if metadata.get("init_geometry_source") == "sam3d_body_prediction" or any(str(value).startswith("sam3d_body") for value in sources if value is not None):
        return True
    if metadata.get("mhr_init_files") or metadata.get("mhr_init_metadata"):
        return True
    return False


def validate_mhr_init_root_metadata(metadata: Mapping[str, Any], label: str, *, required: bool | None = None) -> dict[str, str]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{label} metadata must be a mapping, got {type(metadata).__name__}")
    if required is None:
        required = mhr_init_root_revision_required(metadata)
    expected = mhr_init_root_metadata()
    present = any(key in metadata for key in expected)
    if not required and not present:
        return {}
    mismatched = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatched:
        raise ValueError(f"{label} has stale MHR initialization root metadata: {mismatched}")
    return expected


def validate_mhr_init_translation_metadata(metadata: Mapping[str, Any], label: str, *, required: bool = False) -> dict[str, str]:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{label} metadata must be a mapping, got {type(metadata).__name__}")
    expected = mhr_init_translation_metadata()
    present = any(key in metadata for key in expected)
    if not required and not present:
        return {}
    mismatched = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatched:
        raise ValueError(f"{label} has stale MHR initialization translation metadata: {mismatched}")
    return expected


def mhr_init_translation_is_decode_consistent(metadata: Mapping[str, Any], label: str = "MHR initialization") -> bool:
    if validate_mhr_init_translation_metadata(metadata, label):
        return True
    if any(isinstance(metadata.get(key), Mapping) for key in ("mhr_refit", "mhr_full_refit", "mhr_translation_uniform_scale_refit")):
        return True
    sources = (metadata.get("source"), metadata.get("mhr_init_source"))
    return any(isinstance(source, str) and source.startswith("sam3d_body_rgb_mhr_") and source.endswith("_refit_aligned_to_metric_depth") for source in sources)


def merge_mhr_init_root_metadata(metadata_by_view: Any, label: str) -> dict[str, str]:
    values = list(metadata_by_view)
    if not values:
        raise ValueError(f"{label} requires at least one MHR initialization view")
    for index, metadata in enumerate(values):
        validate_mhr_init_root_metadata(metadata, f"{label} view {index}", required=True)
    return mhr_init_root_metadata()


def merge_mhr_init_translation_metadata(metadata_by_view: Any, label: str) -> dict[str, str]:
    values = list(metadata_by_view)
    if not values:
        raise ValueError(f"{label} requires at least one MHR initialization view")
    consistent = [mhr_init_translation_is_decode_consistent(metadata, f"{label} view {index}") for index, metadata in enumerate(values)]
    return mhr_init_translation_metadata() if all(consistent) else {}


def camera_transform_from_record(record: Mapping[str, Any], camera_id: int) -> np.ndarray:
    rotations = np.asarray(record["rot"], dtype=np.float32)
    translations = np.asarray(record["trans"], dtype=np.float32)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or translations.shape != (len(rotations), 3):
        raise ValueError(f"camera record must contain rotations [K, 3, 3] and translations [K, 3], got {rotations.shape} and {translations.shape}")
    transform_index = int(camera_id)
    if "kids" in record:
        camera_ids = np.asarray(record["kids"]).reshape(-1)
        matches = np.flatnonzero(camera_ids == camera_id)
        if len(camera_ids) != len(rotations) or len(matches) != 1:
            raise KeyError(f"camera {camera_id} has {len(matches)} transforms in camera IDs {camera_ids.tolist()}")
        transform_index = int(matches[0])
    if transform_index < 0 or transform_index >= len(rotations):
        raise IndexError(f"camera {camera_id} is outside [0, {len(rotations)})")
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotations[transform_index]
    transform[:3, 3] = translations[transform_index]
    if not np.isfinite(transform).all():
        raise ValueError(f"camera {camera_id} transform contains nonfinite values")
    return transform


def transform_points_to_camera(points: Any, world_to_camera: Any) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float32)
    transform = np.asarray(world_to_camera, dtype=np.float32)
    if points_array.ndim < 1 or points_array.shape[-1] != 3 or transform.shape != (4, 4):
        raise ValueError(f"points must end in 3 and world_to_camera must be [4, 4], got {points_array.shape} and {transform.shape}")
    if not np.isfinite(points_array).all() or not np.isfinite(transform).all():
        raise ValueError("points and world_to_camera must contain only finite values")
    return (points_array @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32, copy=False)


def _proper_rotation(transform: Any, label: str) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape == (4, 4):
        matrix = matrix[:3, :3]
    if matrix.shape != (3, 3):
        raise ValueError(f"{label} must have shape [3,3] or [4,4], got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} must contain only finite values")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5):
        raise ValueError(f"{label} must contain a proper rotation")
    return matrix


def _camera_to_world_rotation(camera_to_world: Any) -> np.ndarray:
    return _proper_rotation(camera_to_world, "camera_to_world")


def mhr_root_rot6d_between_frames(stored_source_rot6d: Any, source_to_target: Any) -> np.ndarray:
    packed = np.asarray(stored_source_rot6d)
    if packed.ndim < 1 or packed.shape[-1] != 6:
        raise ValueError(f"stored_source_rot6d must end in dimension 6, got {packed.shape}")
    if not np.isfinite(packed).all():
        raise ValueError("stored_source_rot6d must contain only finite values")
    frame_rotation = _proper_rotation(source_to_target, "source_to_target")
    stored_source_rotation = rot6d_to_rotmat_np(packed)
    source_euler = Rotation.from_matrix(stored_source_rotation.reshape(-1, 3, 3)).as_euler("ZYX")
    source_geometry_rotation = Rotation.from_euler("xyz", source_euler).as_matrix()
    target_geometry_rotation = np.einsum("ij,bjk->bik", CAMERA_AXIS_FLIP @ frame_rotation @ CAMERA_AXIS_FLIP, source_geometry_rotation)
    target_euler = Rotation.from_matrix(target_geometry_rotation).as_euler("xyz")
    stored_target_rotation = Rotation.from_euler("ZYX", target_euler).as_matrix()
    return rotmat_to_6d_np(stored_target_rotation).reshape(*packed.shape[:-1], 6).astype(np.float32)


def mhr_translation_between_frames(stored_source_translation: Any, source_root_joint: Any, source_to_target: Any) -> np.ndarray:
    translation = np.asarray(stored_source_translation)
    root_joint = np.asarray(source_root_joint)
    transform = np.asarray(source_to_target, dtype=np.float64)
    if translation.ndim < 1 or translation.shape[-1] != 3 or root_joint.shape != translation.shape:
        raise ValueError(f"stored_source_translation and source_root_joint must share a shape ending in 3, got {translation.shape} and {root_joint.shape}")
    if not np.isfinite(translation).all() or not np.isfinite(root_joint).all():
        raise ValueError("stored_source_translation and source_root_joint must contain only finite values")
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"source_to_target must be a finite rigid [4,4] transform, got {transform.shape}")
    rotation = _proper_rotation(transform, "source_to_target")
    root_offset = root_joint.astype(np.float64) - translation.astype(np.float64)
    transformed = translation.astype(np.float64) @ rotation.T + transform[:3, 3] + root_offset @ rotation.T - root_offset
    return transformed.astype(translation.dtype, copy=False)


def repair_legacy_mhr_world_translation(legacy_world_translation: Any, world_root_joint: Any, camera_to_world: Any) -> np.ndarray:
    camera_to_world = np.asarray(camera_to_world, dtype=np.float64)
    if camera_to_world.shape != (4, 4) or not np.isfinite(camera_to_world).all() or not np.allclose(camera_to_world[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"camera_to_world must be a finite rigid [4,4] transform, got {camera_to_world.shape}")
    _proper_rotation(camera_to_world, "camera_to_world")
    world_to_camera = np.linalg.inv(camera_to_world)
    translation_camera = transform_points_to_camera(legacy_world_translation, world_to_camera)
    root_joint_camera = transform_points_to_camera(world_root_joint, world_to_camera)
    return mhr_translation_between_frames(translation_camera, root_joint_camera, camera_to_world)


def sam3d_root_camera_to_world_rot6d(global_rot_zyx: Any, camera_to_world: Any) -> np.ndarray:
    euler = np.asarray(global_rot_zyx, dtype=np.float64)
    if euler.ndim < 1 or euler.shape[-1] != 3:
        raise ValueError(f"global_rot_zyx must end in dimension 3, got {euler.shape}")
    if not np.isfinite(euler).all():
        raise ValueError("global_rot_zyx must contain only finite values")
    camera_rotation = _camera_to_world_rotation(camera_to_world)
    camera_geometry_rotation = Rotation.from_euler("xyz", euler.reshape(-1, 3)).as_matrix()
    world_geometry_rotation = np.einsum("ij,bjk->bik", CAMERA_AXIS_FLIP @ camera_rotation @ CAMERA_AXIS_FLIP, camera_geometry_rotation)
    world_euler = Rotation.from_matrix(world_geometry_rotation).as_euler("xyz")
    stored_world_rotation = Rotation.from_euler("ZYX", world_euler).as_matrix()
    return rotmat_to_6d_np(stored_world_rotation).reshape(*euler.shape[:-1], 6).astype(np.float32)


def repair_legacy_sam3d_world_root_rot6d(legacy_world_rot6d: Any, camera_to_world: Any) -> np.ndarray:
    packed = np.asarray(legacy_world_rot6d)
    if packed.ndim < 1 or packed.shape[-1] != 6:
        raise ValueError(f"legacy_world_rot6d must end in dimension 6, got {packed.shape}")
    if not np.isfinite(packed).all():
        raise ValueError("legacy_world_rot6d must contain only finite values")
    camera_rotation = _camera_to_world_rotation(camera_to_world)
    legacy_world_rotation = rot6d_to_rotmat_np(packed)
    camera_stored_rotation = np.einsum("ij,...jk->...ik", camera_rotation.T, legacy_world_rotation)
    camera_euler = Rotation.from_matrix(camera_stored_rotation.reshape(-1, 3, 3)).as_euler("ZYX").reshape(*packed.shape[:-1], 3)
    return sam3d_root_camera_to_world_rot6d(camera_euler, camera_rotation)
