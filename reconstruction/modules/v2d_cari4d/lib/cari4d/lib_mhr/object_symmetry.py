from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


OBJECT_SYMMETRY_REVISION = "object-symmetry-v2-group-closure"
OBJECT_SYMMETRY_MODE_FINITE = 0
OBJECT_SYMMETRY_MODE_FULL_SO3 = 1
OBJECT_SYMMETRY_MODE_NAMES = {
    OBJECT_SYMMETRY_MODE_FINITE: "finite",
    OBJECT_SYMMETRY_MODE_FULL_SO3: "full_so3",
}


@dataclass(frozen=True)
class ObjectSymmetry:
    transforms: np.ndarray
    mode: int = OBJECT_SYMMETRY_MODE_FINITE
    center: np.ndarray | None = None

    def __post_init__(self) -> None:
        transforms = np.asarray(self.transforms, dtype=np.float32)
        mode = int(self.mode)
        center = np.zeros(3, dtype=np.float32) if self.center is None else np.asarray(self.center, dtype=np.float32)
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError(f"Object symmetry transforms must have shape [S,4,4], got {transforms.shape}")
        if mode not in OBJECT_SYMMETRY_MODE_NAMES:
            raise ValueError(f"Unsupported object symmetry mode {mode}")
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(f"Object symmetry center must be finite with shape [3], got {center.shape}")
        for index, transform in enumerate(transforms):
            _validate_rigid_transform(transform, f"object symmetry transform {index}")
        if mode == OBJECT_SYMMETRY_MODE_FULL_SO3 and (len(transforms) != 1 or not np.allclose(transforms[0], np.eye(4), rtol=0.0, atol=1e-6)):
            raise ValueError("Full SO(3) symmetry must use one identity finite representative")
        object.__setattr__(self, "transforms", transforms)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "center", center)

    @property
    def mode_name(self) -> str:
        return OBJECT_SYMMETRY_MODE_NAMES[self.mode]

    @property
    def is_full_so3(self) -> bool:
        return self.mode == OBJECT_SYMMETRY_MODE_FULL_SO3


def _as_transform(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (16,):
        arr = arr.reshape(4, 4)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 transform or flat 16-vector, got {arr.shape}")
    _validate_rigid_transform(arr, name)
    return arr


def _validate_rigid_transform(transform: np.ndarray, name: str) -> None:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 transform, got {transform.shape}")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"{name} has an invalid rotation")


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError("continuous symmetry axis must be non-zero")
    x, y, z = axis / norm
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _canonical_axis_line(axis: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    offset = np.asarray(offset, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(axis).all() or not np.isfinite(offset).all() or norm <= 0:
        raise ValueError("Continuous symmetry axis and offset must be finite and the axis must be non-zero")
    axis = axis / norm
    first_nonzero = np.flatnonzero(np.abs(axis) > 1e-12)
    if len(first_nonzero) and axis[first_nonzero[0]] < 0:
        axis = -axis
    point = offset - axis * float(np.dot(axis, offset))
    return axis, point


def _axis_lines(info: Mapping[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for item in info.get("symmetries_continuous") or []:
        axis, point = _canonical_axis_line(item["axis"], item.get("offset", [0.0, 0.0, 0.0]))
        duplicate = False
        for existing_axis, existing_point in lines:
            if abs(float(np.dot(axis, existing_axis))) >= 1.0 - 1e-8:
                line_distance = np.linalg.norm((point - existing_point) - existing_axis * float(np.dot(existing_axis, point - existing_point)))
                if line_distance <= 1e-6:
                    duplicate = True
                    break
        if not duplicate:
            lines.append((axis, point))
    return lines


def _common_axis_center(lines: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if not lines:
        return np.zeros(3, dtype=np.float64)
    matrices = [np.eye(3) - axis[:, None] @ axis[None] for axis, _ in lines]
    system = np.concatenate(matrices, axis=0)
    target = np.concatenate([matrix @ point for matrix, (_, point) in zip(matrices, lines)], axis=0)
    center, _, _, _ = np.linalg.lstsq(system, target, rcond=None)
    residuals = [np.linalg.norm(matrix @ (center - point)) for matrix, (_, point) in zip(matrices, lines)]
    if max(residuals, default=0.0) > 1e-5:
        raise ValueError(f"Continuous symmetry axes do not share a common center; maximum line residual is {max(residuals):.6g} m")
    return center


def _classify_axis_lines(lines: list[tuple[np.ndarray, np.ndarray]]) -> tuple[int, np.ndarray, np.ndarray | None]:
    if not lines:
        return OBJECT_SYMMETRY_MODE_FINITE, np.zeros(3, dtype=np.float64), None
    reference_axis, reference_point = lines[0]
    has_nonparallel = False
    for axis, point in lines[1:]:
        if abs(float(np.dot(reference_axis, axis))) < 1.0 - 1e-8:
            has_nonparallel = True
            continue
        line_distance = np.linalg.norm((point - reference_point) - reference_axis * float(np.dot(reference_axis, point - reference_point)))
        if line_distance > 1e-6:
            raise ValueError("Parallel continuous symmetry axes lie on different lines and cannot describe a bounded rigid object")
    center = _common_axis_center(lines)
    if has_nonparallel:
        return OBJECT_SYMMETRY_MODE_FULL_SO3, center, None
    return OBJECT_SYMMETRY_MODE_FINITE, center, reference_axis


def _axis_symmetry_tfs(axis: np.ndarray, center: np.ndarray, angle_degrees: float) -> list[np.ndarray]:
    angle_degrees = float(angle_degrees)
    if not np.isfinite(angle_degrees) or angle_degrees <= 0.0 or angle_degrees > 360.0:
        raise ValueError(f"continuous_angle_degrees must be finite in (0, 360], got {angle_degrees}")
    transforms: list[np.ndarray] = []
    for angle in np.arange(0.0, 360.0, angle_degrees, dtype=np.float64):
        transform = np.eye(4, dtype=np.float64)
        rotation = _axis_rotation(axis, np.deg2rad(angle))
        transform[:3, :3] = rotation
        transform[:3, 3] = center - rotation @ center
        transforms.append(transform)
    return transforms


def _deduplicate_transforms(transforms: list[np.ndarray], decimals: int = 10) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    scale = 10 ** decimals
    for transform in transforms:
        key = np.rint(transform * scale).astype(np.int64).tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(transform)
    return unique


def _finite_group_closure(generators: list[np.ndarray], max_transforms: int = 4096) -> list[np.ndarray]:
    identity = np.eye(4, dtype=np.float64)
    generator_set = _deduplicate_transforms([identity, *generators, *[np.linalg.inv(transform) for transform in generators]], decimals=8)
    closure = [identity]
    seen = {np.rint(identity * 1e8).astype(np.int64).tobytes()}
    index = 0
    while index < len(closure):
        current = closure[index]
        index += 1
        for generator in generator_set:
            product = current @ generator
            _validate_rigid_transform(product, "discrete symmetry group product")
            key = np.rint(product * 1e8).astype(np.int64).tobytes()
            if key in seen:
                continue
            seen.add(key)
            closure.append(product)
            if len(closure) > max_transforms:
                raise ValueError(f"Discrete symmetry generators exceed the bounded closure size of {max_transforms}")
    return closure


def _transform_axis_line(transform: np.ndarray, line: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    axis, point = line
    transformed_axis = transform[:3, :3] @ axis
    transformed_point = transform[:3, :3] @ point + transform[:3, 3]
    return _canonical_axis_line(transformed_axis, transformed_point)


def _aligned_object_symmetry(info: Mapping[str, Any], *, include_identity: bool, continuous_angle_degrees: float) -> ObjectSymmetry:
    identity = np.eye(4, dtype=np.float64)
    discrete_generators = [_as_transform(item, name=f"symmetries_discrete[{idx}]") for idx, item in enumerate(info.get("symmetries_discrete") or [])]
    discrete_transforms = _finite_group_closure(discrete_generators)
    source_lines = _axis_lines(info)
    orbit_lines = [_transform_axis_line(transform, line) for transform in discrete_transforms for line in source_lines]
    mode, center, axis = _classify_axis_lines(orbit_lines)
    if mode == OBJECT_SYMMETRY_MODE_FULL_SO3:
        transforms = [identity] if include_identity else []
    elif axis is not None:
        continuous_transforms = _axis_symmetry_tfs(axis, center, continuous_angle_degrees)
        transforms = [continuous @ discrete for discrete in discrete_transforms for continuous in continuous_transforms]
    else:
        transforms = discrete_transforms
    transforms = _deduplicate_transforms(transforms)
    if not include_identity:
        transforms = [transform for transform in transforms if not np.allclose(transform, identity, rtol=0.0, atol=1e-10)]
    if not transforms and mode != OBJECT_SYMMETRY_MODE_FULL_SO3:
        return ObjectSymmetry(np.empty((0, 4, 4), dtype=np.float32), mode=mode, center=center.astype(np.float32))
    finite_representatives = np.stack(transforms, axis=0).astype(np.float32) if transforms else identity[None].astype(np.float32)
    return ObjectSymmetry(finite_representatives, mode=mode, center=center.astype(np.float32))


def object_mesh_uses_symmetry_alignment(object_mesh_path: str | Path) -> bool:
    name = Path(object_mesh_path).name
    if name == "output_aligned.glb":
        return True
    if name == "output.glb":
        return False
    raise ValueError(f"Cannot determine output_symmetry.json frame for object mesh {object_mesh_path}; expected output_aligned.glb or output.glb")


def object_symmetry_frame_for_mesh(object_mesh_path: str | Path) -> str:
    return "output_aligned_mesh_frame" if object_mesh_uses_symmetry_alignment(object_mesh_path) else "source_mesh_frame_from_output_symmetry_alignment"


def resolve_object_symmetry_mesh_path(metadata: Mapping[str, Any], requested_mesh_path: str | Path | None = None) -> Path:
    pose_mesh_value = metadata.get("mhr_contact_source")
    symmetry_mesh_value = metadata.get("object_symmetry_mesh_source")
    pose_mesh_path = Path(str(pose_mesh_value)) if pose_mesh_value else None
    requested_path = Path(requested_mesh_path) if requested_mesh_path is not None else None
    if pose_mesh_path is not None and requested_path is not None:
        pose_frame = object_symmetry_frame_for_mesh(pose_mesh_path)
        requested_frame = object_symmetry_frame_for_mesh(requested_path)
        if pose_frame != requested_frame:
            raise ValueError(f"Explicit object symmetry mesh frame {requested_frame} differs from mhr_contact_source object pose frame {pose_frame}")
    resolved = pose_mesh_path or requested_path or (Path(str(symmetry_mesh_value)) if symmetry_mesh_value else None)
    if resolved is None:
        raise ValueError("Cannot determine object symmetry mesh frame; metadata has no mhr_contact_source or object_symmetry_mesh_source and no mesh was requested")
    object_symmetry_frame_for_mesh(resolved)
    return resolved


def validate_object_symmetry_metadata(metadata: Mapping[str, Any]) -> None:
    pose_mesh_value = metadata.get("mhr_contact_source")
    symmetry_mesh_value = metadata.get("object_symmetry_mesh_source")
    frame_value = metadata.get("object_symmetry_frame")
    revision = metadata.get("object_symmetry_revision")
    mode = metadata.get("object_symmetry_mode")
    center = metadata.get("object_symmetry_center")
    descriptor_values = (revision, mode, center)
    if any(value is not None for value in descriptor_values):
        if revision != OBJECT_SYMMETRY_REVISION:
            raise ValueError(f"Recorded object_symmetry_revision is stale: {revision!r}")
        if mode not in OBJECT_SYMMETRY_MODE_NAMES.values():
            raise ValueError(f"Recorded object_symmetry_mode is unsupported: {mode!r}")
        center_array = np.asarray(center, dtype=np.float64)
        if center_array.shape != (3,) or not np.isfinite(center_array).all():
            raise ValueError(f"Recorded object_symmetry_center must be finite with shape [3], got {center_array.shape}")
    if not symmetry_mesh_value and not frame_value:
        return
    resolved = resolve_object_symmetry_mesh_path(metadata)
    resolved_frame = object_symmetry_frame_for_mesh(resolved)
    if pose_mesh_value and symmetry_mesh_value:
        symmetry_frame = object_symmetry_frame_for_mesh(symmetry_mesh_value)
        if symmetry_frame != resolved_frame:
            raise ValueError(f"Recorded object symmetry mesh frame {symmetry_frame} differs from mhr_contact_source object pose frame {resolved_frame}")
    if frame_value and str(frame_value) != resolved_frame:
        raise ValueError(f"Recorded object_symmetry_frame {frame_value} differs from mhr_contact_source object pose frame {resolved_frame}")


def output_symmetry_alignment_matrix(info: Mapping[str, Any]) -> np.ndarray:
    """Return the output_symmetry.json alignment transform.

    The flat export symmetry file stores symmetries in its alignment/OBB frame.
    This transform maps mesh coordinates into that alignment frame as
    ``p_aligned = R @ (p_mesh - centroid)``.
    """

    alignment = info.get("alignment")
    if not isinstance(alignment, Mapping):
        raise ValueError("output symmetry info is missing an alignment block")
    matrix = _as_transform(alignment["rotation"], name="alignment.rotation")
    matrix = matrix.copy()
    centroid = np.asarray(alignment.get("centroid", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    matrix[:3, 3] = -matrix[:3, :3] @ centroid
    return matrix


def output_object_symmetry(
    info: Mapping[str, Any],
    *,
    include_identity: bool = True,
    to_mesh_frame: bool = True,
    continuous_angle_degrees: float = 5.0,
) -> ObjectSymmetry:
    """Parse output_symmetry.json metadata into exact group semantics."""

    symmetry = _aligned_object_symmetry(info, include_identity=include_identity, continuous_angle_degrees=continuous_angle_degrees)
    transforms = symmetry.transforms.astype(np.float64)
    center = symmetry.center.astype(np.float64)
    if to_mesh_frame:
        alignment = output_symmetry_alignment_matrix(info)
        alignment_inv = np.linalg.inv(alignment)
        transforms = alignment_inv[None] @ transforms @ alignment[None]
        center = alignment_inv[:3, :3] @ center + alignment_inv[:3, 3]
    if include_identity and len(transforms):
        transforms[0] = np.eye(4, dtype=np.float64)
    return ObjectSymmetry(transforms.astype(np.float32), mode=symmetry.mode, center=center.astype(np.float32))


def bop_object_symmetry(info: Mapping[str, Any], *, continuous_angle_degrees: float = 5.0) -> ObjectSymmetry:
    """Parse BOP model-info symmetry metadata into meter-based group semantics."""

    normalized = dict(info)
    normalized["symmetries_discrete"] = []
    for index, item in enumerate(info.get("symmetries_discrete") or []):
        transform = _as_transform(item, name=f"symmetries_discrete[{index}]").copy()
        transform[:3, 3] *= 0.001
        normalized["symmetries_discrete"].append(transform.reshape(-1).tolist())
    normalized["symmetries_continuous"] = []
    for item in info.get("symmetries_continuous") or []:
        converted = dict(item)
        converted["offset"] = (np.asarray(item.get("offset", [0.0, 0.0, 0.0]), dtype=np.float64) * 0.001).tolist()
        normalized["symmetries_continuous"].append(converted)
    return output_object_symmetry(normalized, to_mesh_frame=False, continuous_angle_degrees=continuous_angle_degrees)


def output_symmetry_tfs(
    info: Mapping[str, Any],
    *,
    include_identity: bool = True,
    to_mesh_frame: bool = True,
    continuous_angle_degrees: float = 5.0,
) -> np.ndarray:
    """Return finite symmetry transforms, rejecting non-finite symmetry groups."""

    symmetry = output_object_symmetry(info, include_identity=include_identity, to_mesh_frame=to_mesh_frame, continuous_angle_degrees=continuous_angle_degrees)
    if symmetry.is_full_so3:
        raise ValueError("Multiple independent continuous symmetry axes generate full SO(3); use output_object_symmetry() so the mode and center are preserved")
    return symmetry.transforms


def load_output_symmetry_tfs(
    path: str | Path,
    *,
    object_mesh_path: str | Path,
    include_identity: bool = True,
    continuous_angle_degrees: float = 5.0,
) -> np.ndarray:
    to_mesh_frame = not object_mesh_uses_symmetry_alignment(object_mesh_path)
    with Path(path).open() as f:
        info = json.load(f)
    return output_symmetry_tfs(
        info,
        include_identity=include_identity,
        to_mesh_frame=to_mesh_frame,
        continuous_angle_degrees=continuous_angle_degrees,
    )


def load_output_object_symmetry(
    path: str | Path,
    *,
    object_mesh_path: str | Path,
    include_identity: bool = True,
    continuous_angle_degrees: float = 5.0,
) -> ObjectSymmetry:
    to_mesh_frame = not object_mesh_uses_symmetry_alignment(object_mesh_path)
    with Path(path).open() as file:
        info = json.load(file)
    return output_object_symmetry(info, include_identity=include_identity, to_mesh_frame=to_mesh_frame, continuous_angle_degrees=continuous_angle_degrees)


def load_optional_output_symmetry_tfs(path: str | Path | None, *, object_mesh_path: str | Path | None) -> np.ndarray | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    if object_mesh_path is None:
        raise ValueError(f"Object mesh path is required when loading {path}")
    return load_output_symmetry_tfs(path, object_mesh_path=object_mesh_path)


def load_optional_output_object_symmetry(path: str | Path | None, *, object_mesh_path: str | Path | None) -> ObjectSymmetry | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    if object_mesh_path is None:
        raise ValueError(f"Object mesh path is required when loading {path}")
    return load_output_object_symmetry(path, object_mesh_path=object_mesh_path)
