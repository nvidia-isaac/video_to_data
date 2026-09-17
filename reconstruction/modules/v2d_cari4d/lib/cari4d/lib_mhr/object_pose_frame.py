from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contact import BEHAVE_OBJECT_MESH_POSE_FRAME_REVISION, validate_object_mesh_to_pose_transform
from .object_symmetry import output_symmetry_alignment_matrix


OBJECT_POSE_FRAME = "centered_axis_aligned"
OBJECT_POSE_FRAME_REVISION = "cari4d.object_pose_frame.centered_axis_aligned.v1"
OBJECT_POSE_STORAGE_FRAME_KEY = "object_pose_storage_frame"
OBJECT_POSE_STORAGE_TO_TRAINING_KEY = "object_pose_storage_to_training_transform"
OBJECT_MESH_TO_TRAINING_KEY = "object_mesh_to_training_transform"


@dataclass(frozen=True)
class ObjectPoseFrame:
    storage_frame: str
    storage_to_training: np.ndarray
    mesh_to_storage: np.ndarray
    mesh_to_training: np.ndarray

    def __post_init__(self) -> None:
        storage_to_training = validate_object_mesh_to_pose_transform(self.storage_to_training, OBJECT_POSE_STORAGE_TO_TRAINING_KEY)
        mesh_to_storage = validate_object_mesh_to_pose_transform(self.mesh_to_storage, "object_mesh_to_pose_transform")
        mesh_to_training = validate_object_mesh_to_pose_transform(self.mesh_to_training, OBJECT_MESH_TO_TRAINING_KEY)
        expected = storage_to_training @ mesh_to_storage
        if not np.allclose(mesh_to_training, expected, rtol=0.0, atol=1e-6):
            raise ValueError(f"{OBJECT_MESH_TO_TRAINING_KEY} does not equal {OBJECT_POSE_STORAGE_TO_TRAINING_KEY} @ object_mesh_to_pose_transform")
        object.__setattr__(self, "storage_frame", str(self.storage_frame))
        object.__setattr__(self, "storage_to_training", storage_to_training)
        object.__setattr__(self, "mesh_to_storage", mesh_to_storage)
        object.__setattr__(self, "mesh_to_training", mesh_to_training)

    def metadata(self) -> dict[str, Any]:
        return {
            "object_pose_frame": OBJECT_POSE_FRAME,
            "object_pose_frame_revision": OBJECT_POSE_FRAME_REVISION,
            OBJECT_POSE_STORAGE_FRAME_KEY: self.storage_frame,
            OBJECT_POSE_STORAGE_TO_TRAINING_KEY: self.storage_to_training.tolist(),
            OBJECT_MESH_TO_TRAINING_KEY: self.mesh_to_training.tolist(),
        }


def _identity() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def _object_mesh_source(metadata: Mapping[str, Any]) -> Path | None:
    value = metadata.get("object_symmetry_mesh_source") or metadata.get("mhr_contact_source") or metadata.get("object_mesh_file")
    return None if value is None else Path(str(value))


def _load_output_alignment(metadata: Mapping[str, Any]) -> np.ndarray:
    source = metadata.get("object_symmetry_source")
    if source is None:
        raise ValueError("Source-frame object poses require object_symmetry_source so the centered, axis-aligned training frame can be reconstructed")
    path = Path(str(source))
    if not path.is_file():
        raise FileNotFoundError(f"Object symmetry alignment metadata does not exist: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, Mapping):
        raise TypeError(f"Object symmetry alignment metadata must decode to an object: {path}")
    return output_symmetry_alignment_matrix(payload).astype(np.float32)


def resolve_object_pose_frame(metadata: Mapping[str, Any], *, assume_aligned_without_mesh: bool = False) -> ObjectPoseFrame:
    metadata = dict(metadata)
    mesh_to_storage = validate_object_mesh_to_pose_transform(metadata.get("object_mesh_to_pose_transform", _identity()))
    revision = metadata.get("object_pose_frame_revision")
    if revision is not None:
        if revision != OBJECT_POSE_FRAME_REVISION or metadata.get("object_pose_frame") != OBJECT_POSE_FRAME:
            raise ValueError(f"Unsupported object pose frame metadata: frame={metadata.get('object_pose_frame')!r}, revision={revision!r}")
        storage_to_training = validate_object_mesh_to_pose_transform(metadata.get(OBJECT_POSE_STORAGE_TO_TRAINING_KEY), OBJECT_POSE_STORAGE_TO_TRAINING_KEY)
        mesh_to_training = validate_object_mesh_to_pose_transform(metadata.get(OBJECT_MESH_TO_TRAINING_KEY), OBJECT_MESH_TO_TRAINING_KEY)
        return ObjectPoseFrame(str(metadata.get(OBJECT_POSE_STORAGE_FRAME_KEY, "")), storage_to_training, mesh_to_storage, mesh_to_training)

    symmetry_frame = metadata.get("object_symmetry_frame")
    mesh_source = _object_mesh_source(metadata)
    if symmetry_frame == "source_mesh_frame_from_output_symmetry_alignment":
        storage_frame = str(symmetry_frame)
        storage_to_training = _load_output_alignment(metadata)
    elif symmetry_frame == "output_aligned_mesh_frame" or mesh_source is not None and mesh_source.name == "output_aligned.glb":
        storage_frame = "output_aligned_mesh_frame"
        storage_to_training = _identity()
    elif metadata.get("object_mesh_pose_frame_revision") == BEHAVE_OBJECT_MESH_POSE_FRAME_REVISION:
        storage_frame = "behave_registration_template_frame"
        storage_to_training = _identity()
    elif metadata.get("source") == "behave_native_mhr_validation" and mesh_source is not None:
        raise ValueError("stale object-mesh pose-frame metadata; regenerate the BEHAVE packed file with its registration-template transform")
    elif assume_aligned_without_mesh and mesh_source is None:
        storage_frame = OBJECT_POSE_FRAME
        storage_to_training = _identity()
    else:
        raise ValueError("Object pose storage frame is ambiguous; provide current object-pose-frame metadata or an aligned object mesh")
    return ObjectPoseFrame(storage_frame, storage_to_training, mesh_to_storage, storage_to_training @ mesh_to_storage)


def stamp_object_pose_frame_metadata(metadata: Mapping[str, Any], *, assume_aligned_without_mesh: bool = False) -> dict[str, Any]:
    stamped = dict(metadata)
    stamped.update(resolve_object_pose_frame(stamped, assume_aligned_without_mesh=assume_aligned_without_mesh).metadata())
    return stamped


def object_poses_to_training_frame(poses: Any, storage_to_training: Any) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32)
    transform = validate_object_mesh_to_pose_transform(storage_to_training, OBJECT_POSE_STORAGE_TO_TRAINING_KEY)
    if poses.shape[-2:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"Object poses must be finite [...,4,4], got {poses.shape}")
    return (poses @ np.linalg.inv(transform).astype(np.float32)).astype(np.float32)


def object_rotations_translations_to_training_frame(rotations: Any, translations: Any, storage_to_training: Any) -> tuple[np.ndarray, np.ndarray]:
    rotations = np.asarray(rotations, dtype=np.float32)
    translations = np.asarray(translations, dtype=np.float32)
    if rotations.shape[:-2] != translations.shape[:-1] or rotations.shape[-2:] != (3, 3) or translations.shape[-1:] != (3,):
        raise ValueError(f"Object rotation/translation shapes are incompatible: {rotations.shape} and {translations.shape}")
    poses = np.broadcast_to(_identity(), rotations.shape[:-2] + (4, 4)).copy()
    poses[..., :3, :3] = rotations
    poses[..., :3, 3] = translations
    aligned = object_poses_to_training_frame(poses, storage_to_training)
    return aligned[..., :3, :3].copy(), aligned[..., :3, 3].copy()


def object_symmetry_to_training_frame(transforms: Any, center: Any, storage_to_training: Any) -> tuple[np.ndarray, np.ndarray]:
    transforms = np.asarray(transforms, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    alignment = validate_object_mesh_to_pose_transform(storage_to_training, OBJECT_POSE_STORAGE_TO_TRAINING_KEY)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or not np.isfinite(transforms).all():
        raise ValueError(f"Object symmetry transforms must be finite [S,4,4], got {transforms.shape}")
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError(f"Object symmetry center must be finite [3], got {center.shape}")
    inverse = np.linalg.inv(alignment).astype(np.float32)
    transformed = alignment[None] @ transforms @ inverse[None]
    transformed_center = alignment[:3, :3] @ center + alignment[:3, 3]
    return transformed.astype(np.float32), transformed_center.astype(np.float32)
