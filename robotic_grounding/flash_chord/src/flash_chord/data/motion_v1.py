# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the versioned ``motion_v1/single_robot`` reference schema."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from flash_chord.assets.registry import resolve_object_asset_specs
from flash_chord.data.parquet import read_parquet_row
from flash_chord.data.reference import ObjectAssetSpec, ReferenceMetadata, reference_frame_slice
from flash_chord.data.resampling import lerp, nearest, playback_times, slerp_tracks_wxyz

_COLUMNS = (
    "schema_version",
    "motion_kind",
    "source_dataset",
    "raw_motion_file",
    "fps",
    "coord_frame",
    "robot_joint_names",
    "robot_root_position",
    "robot_root_wxyz",
    "robot_joint_positions",
    "ee_link_names",
    "ee_pose_w",
    "hand_sides",
    "hand_frame_names",
    "hand_frames_w",
    "object_name",
    "object_body_names",
    "object_mesh_paths",
    "object_urdf_paths",
    "object_mesh_radius",
    "object_articulation",
    "object_body_position",
    "object_body_wxyz",
    "hand_object_contact_positions",
    "hand_object_contact_normals",
    "hand_object_contact_part_ids",
    "hand_contact_active",
    "sequence_id",
    "robot_name",
)


@dataclass
class MotionV1Reference:
    """Validated host trajectories for one named whole-robot motion."""

    _metadata: ReferenceMetadata
    _fps: float
    _sides: tuple[str, ...]
    _robot_joint_names: list[str]
    _robot_joint_pos: np.ndarray
    _robot_root_pos_w: np.ndarray
    _robot_root_quat_w: np.ndarray
    _robot_frame_names: list[str]
    _robot_frame_pos_w: np.ndarray
    _robot_frame_quat_w: np.ndarray
    _object_body_pos_w: np.ndarray
    _object_body_quat_w: np.ndarray
    _object_articulation: np.ndarray
    _contact_pos_w: dict[str, np.ndarray]
    _contact_normal_w: dict[str, np.ndarray]
    _contact_part_ids: dict[str, np.ndarray]
    _contact_active: dict[str, np.ndarray]
    _object_name: str
    _object_body_names: list[str]
    _object_mesh_paths: list[str]
    _object_urdf_paths: list[str]
    _object_mesh_radius: np.ndarray
    _object_assets: tuple[ObjectAssetSpec, ...]

    @property
    def num_frames(self) -> int:
        return self._robot_joint_pos.shape[0]

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def metadata(self) -> ReferenceMetadata:
        return self._metadata

    @property
    def sides(self) -> tuple[str, ...]:
        return self._sides

    def robot_joint_pos(self) -> np.ndarray:
        return self._robot_joint_pos

    def robot_joint_names(self) -> list[str]:
        return self._robot_joint_names

    def robot_root_pos_w(self) -> np.ndarray:
        return self._robot_root_pos_w

    def robot_root_quat_w(self) -> np.ndarray:
        return self._robot_root_quat_w

    def robot_frame_names(self) -> list[str]:
        return self._robot_frame_names

    def robot_frame_pos_w(self) -> np.ndarray:
        return self._robot_frame_pos_w

    def robot_frame_quat_w(self) -> np.ndarray:
        return self._robot_frame_quat_w

    def object_body_pos_w(self) -> np.ndarray:
        return self._object_body_pos_w

    def object_body_quat_w(self) -> np.ndarray:
        return self._object_body_quat_w

    def object_articulation(self) -> np.ndarray:
        return self._object_articulation

    def contact_pos_w(self, side: str) -> np.ndarray:
        return self._contact_pos_w[side]

    def contact_normal_w(self, side: str) -> np.ndarray:
        return self._contact_normal_w[side]

    def contact_part_ids(self, side: str) -> np.ndarray:
        return self._contact_part_ids[side]

    def contact_active(self, side: str) -> np.ndarray:
        return self._contact_active[side]

    def object_name(self) -> str:
        return self._object_name

    def object_body_names(self) -> list[str]:
        return self._object_body_names

    def object_mesh_paths(self) -> list[str]:
        return self._object_mesh_paths

    def object_urdf_paths(self) -> list[str]:
        return self._object_urdf_paths

    def object_mesh_radius(self) -> np.ndarray:
        return self._object_mesh_radius

    def object_assets(self) -> tuple[ObjectAssetSpec, ...]:
        return self._object_assets

    def frame_window(self, start_frame: int = 0, end_frame: int = -1) -> MotionV1Reference:
        """Return a copy restricted to the half-open source-frame window."""
        frames = reference_frame_slice(self.num_frames, start_frame, end_frame)
        if frames.start == 0 and frames.stop == self.num_frames:
            return self

        def sliced(values: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(values[frames])

        return replace(
            self,
            _robot_joint_pos=sliced(self._robot_joint_pos),
            _robot_root_pos_w=sliced(self._robot_root_pos_w),
            _robot_root_quat_w=sliced(self._robot_root_quat_w),
            _robot_frame_pos_w=sliced(self._robot_frame_pos_w),
            _robot_frame_quat_w=sliced(self._robot_frame_quat_w),
            _object_body_pos_w=sliced(self._object_body_pos_w),
            _object_body_quat_w=sliced(self._object_body_quat_w),
            _object_articulation=sliced(self._object_articulation),
            _contact_pos_w={side: sliced(values) for side, values in self._contact_pos_w.items()},
            _contact_normal_w={side: sliced(values) for side, values in self._contact_normal_w.items()},
            _contact_part_ids={side: sliced(values) for side, values in self._contact_part_ids.items()},
            _contact_active={side: sliced(values) for side, values in self._contact_active.items()},
        )


def load_motion_v1(
    parquet_path: str,
    control_fps: float | None = None,
    motion_speed: float = 1.0,
    source_frame_playback: bool = False,
) -> MotionV1Reference:
    """Load, validate, and optionally resample one ``motion_v1/single_robot`` parquet."""
    if source_frame_playback and control_fps is None:
        raise ValueError("control_fps is required for source-frame playback")
    if source_frame_playback and motion_speed != 1.0:
        raise ValueError("motion_speed must be 1.0 for source-frame playback")
    if control_fps is None and motion_speed != 1.0:
        raise ValueError("control_fps is required when motion_speed is not 1.0")
    row = read_parquet_row(parquet_path, columns=_COLUMNS)
    cells = row.cells

    schema_version = _string(cells, "schema_version")
    if schema_version != "motion_v1":
        raise ValueError(f"expected schema_version='motion_v1', got {schema_version!r}: {row.path}")
    motion_kind = _string(cells, "motion_kind")
    if motion_kind != "single_robot":
        raise ValueError(f"motion_v1 loader supports motion_kind='single_robot', got {motion_kind!r}")
    coord_frame = _string(cells, "coord_frame")
    supported_coord_frames = ("world", "robot_base_z_up")
    if coord_frame not in supported_coord_frames:
        raise ValueError(
            "motion_v1 poses must use a common world-coordinate convention "
            f"from {supported_coord_frames}, got {coord_frame!r}"
        )

    source_fps = _positive_float(cells, "fps")
    joint_names = _unique_names(cells, "robot_joint_names", require_nonempty=True)
    joint_pos = _finite_array(cells, "robot_joint_positions", ndim=2)
    num_frames = joint_pos.shape[0]
    if joint_pos.shape[1] != len(joint_names):
        raise ValueError(
            f"robot_joint_positions name-axis mismatch: shape {joint_pos.shape}, {len(joint_names)} robot_joint_names"
        )
    root_pos = _trajectory(cells, "robot_root_position", num_frames, (3,))
    root_quat = _quaternion_trajectory(cells, "robot_root_wxyz", num_frames, ())

    frame_names = _unique_names(cells, "ee_link_names", require_nonempty=False)
    frame_pose = _finite_array(cells, "ee_pose_w", ndim=3)
    expected_frame_shape = (num_frames, len(frame_names), 7)
    if frame_pose.shape != expected_frame_shape:
        raise ValueError(f"ee_pose_w must have shape {expected_frame_shape}, got {frame_pose.shape}")
    frame_pos = np.ascontiguousarray(frame_pose[..., :3])
    frame_quat = _validated_quaternions(np.ascontiguousarray(frame_pose[..., 3:]), "ee_pose_w")

    sides = tuple(_unique_names(cells, "hand_sides", require_nonempty=False))
    hand_frame_names, hand_frame_pos, hand_frame_quat = _optional_hand_frames(
        cells,
        sides,
        num_frames,
    )
    duplicate_frames = sorted(set(frame_names).intersection(hand_frame_names))
    if duplicate_frames:
        raise ValueError(f"motion_v1 frame names must be globally unique; duplicates: {duplicate_frames}")
    if hand_frame_names:
        frame_names += hand_frame_names
        frame_pos = np.concatenate((frame_pos, hand_frame_pos), axis=1)
        frame_quat = np.concatenate((frame_quat, hand_frame_quat), axis=1)
    object_names = _unique_names(cells, "object_body_names", require_nonempty=True)
    num_object_bodies = len(object_names)
    object_pos = _trajectory(cells, "object_body_position", num_frames, (num_object_bodies, 3))
    object_quat = _quaternion_trajectory(cells, "object_body_wxyz", num_frames, (num_object_bodies,))
    articulation = _articulation(cells.get("object_articulation"), num_frames)
    contact_pos, contact_normal, contact_part_ids = _contacts(cells, sides, num_frames, num_object_bodies)
    contact_active = _contact_activity(cells.get("hand_contact_active"), sides, num_frames, contact_part_ids)
    object_name = _string(cells, "object_name")
    object_mesh_paths = _asset_paths(cells, "object_mesh_paths", num_object_bodies)
    object_urdf_paths = _strings(cells.get("object_urdf_paths"))
    object_mesh_radius = _positive_vector(cells, "object_mesh_radius", num_object_bodies)
    if object_urdf_paths and articulation.shape[1]:
        if articulation.shape[1] != 1 or not np.all(articulation == 0.0):
            raise ValueError(
                "motion_v1 references with independent rigid-object URDFs may contain only one "
                "all-zero articulation placeholder column"
            )
        articulation = np.zeros((num_frames, 0), dtype=articulation.dtype)
    source_dataset = _optional_string(cells.get("source_dataset"))
    object_assets = resolve_object_asset_specs(
        source_dataset=source_dataset,
        object_name=object_name,
        body_names=object_names,
        mesh_paths=object_mesh_paths,
        urdf_paths=object_urdf_paths,
        num_articulations=articulation.shape[1],
        reference_path=row.path,
    )

    source_times = target_times = None
    resample = (
        not source_frame_playback and control_fps is not None and (control_fps != source_fps or motion_speed != 1.0)
    )
    if control_fps is not None and not source_frame_playback:
        source_times, target_times = playback_times(num_frames, source_fps, control_fps, motion_speed)
    if resample:
        assert source_times is not None and target_times is not None
        joint_pos = lerp(joint_pos, source_times, target_times)
        root_pos = lerp(root_pos, source_times, target_times)
        root_quat = slerp_tracks_wxyz(root_quat, source_times, target_times)
        frame_pos = lerp(frame_pos, source_times, target_times)
        frame_quat = slerp_tracks_wxyz(frame_quat, source_times, target_times)
        object_pos = lerp(object_pos, source_times, target_times)
        object_quat = slerp_tracks_wxyz(object_quat, source_times, target_times)
        articulation = (
            lerp(articulation, source_times, target_times)
            if articulation.shape[1]
            else np.zeros((len(target_times), 0), dtype=articulation.dtype)
        )
        contact_pos = {side: nearest(values, source_times, target_times) for side, values in contact_pos.items()}
        contact_normal = {side: nearest(values, source_times, target_times) for side, values in contact_normal.items()}
        contact_part_ids = {
            side: nearest(values, source_times, target_times) for side, values in contact_part_ids.items()
        }
        contact_active = {side: nearest(values, source_times, target_times) for side, values in contact_active.items()}

    return MotionV1Reference(
        _metadata=ReferenceMetadata(
            source_path=str(row.path),
            source_fps=source_fps,
            is_resampled=resample,
            source_frame_playback=source_frame_playback,
            schema_version=schema_version,
            motion_kind=motion_kind,
            source_dataset=source_dataset,
            sequence_id=_optional_string(cells.get("sequence_id")),
            robot_name=_optional_string(cells.get("robot_name")),
            raw_motion_file=_optional_string(cells.get("raw_motion_file")),
            coord_frame=coord_frame,
        ),
        _fps=control_fps if control_fps is not None else source_fps,
        _sides=sides,
        _robot_joint_names=joint_names,
        _robot_joint_pos=joint_pos,
        _robot_root_pos_w=root_pos,
        _robot_root_quat_w=root_quat,
        _robot_frame_names=frame_names,
        _robot_frame_pos_w=frame_pos,
        _robot_frame_quat_w=frame_quat,
        _object_body_pos_w=object_pos,
        _object_body_quat_w=object_quat,
        _object_articulation=articulation,
        _contact_pos_w=contact_pos,
        _contact_normal_w=contact_normal,
        _contact_part_ids=contact_part_ids,
        _contact_active=contact_active,
        _object_name=object_name,
        _object_body_names=object_names,
        _object_mesh_paths=object_mesh_paths,
        _object_urdf_paths=object_urdf_paths,
        _object_mesh_radius=object_mesh_radius,
        _object_assets=object_assets,
    )


def _optional_hand_frames(
    cells,
    sides: tuple[str, ...],
    num_frames: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Flatten optional V2D side-indexed hand frames into the named-frame axis."""
    names_payload = cells.get("hand_frame_names")
    poses_payload = cells.get("hand_frames_w")
    names_empty = _empty_per_side_payload(names_payload, len(sides))
    poses_empty = _empty_per_side_payload(poses_payload, len(sides))
    if names_empty and poses_empty:
        return (
            [],
            np.empty((num_frames, 0, 3), dtype=np.float32),
            np.empty((num_frames, 0, 4), dtype=np.float32),
        )
    if names_empty or poses_empty:
        raise ValueError("motion_v1 hand frame names and poses must be provided together")
    if not sides:
        raise ValueError("motion_v1 hand frames require nonempty hand_sides")

    names_by_side = _per_side_values(cells, "hand_frame_names", sides)
    poses_by_side = _per_side_values(cells, "hand_frames_w", sides)
    frame_names: list[str] = []
    frame_positions: list[np.ndarray] = []
    frame_quaternions: list[np.ndarray] = []
    for side in sides:
        names = _strings(names_by_side[side])
        if not names or len(names) != len(set(names)) or any(not name for name in names):
            raise ValueError(f"motion_v1 hand_frame_names[{side}] must contain unique nonempty names")
        poses = np.asarray(poses_by_side[side])
        expected = (num_frames, len(names), 7)
        if poses.shape != expected or not np.issubdtype(poses.dtype, np.number) or not np.all(np.isfinite(poses)):
            raise ValueError(
                f"motion_v1 hand_frames_w[{side}] must contain finite poses with shape {expected}, got {poses.shape}"
            )
        frame_names.extend(names)
        frame_positions.append(np.ascontiguousarray(poses[..., :3]))
        frame_quaternions.append(
            _validated_quaternions(
                np.ascontiguousarray(poses[..., 3:]),
                f"hand_frames_w[{side}]",
            )
        )
    if len(frame_names) != len(set(frame_names)):
        raise ValueError("motion_v1 hand frame names must be unique across hand sides")
    return (
        frame_names,
        np.concatenate(frame_positions, axis=1),
        np.concatenate(frame_quaternions, axis=1),
    )


def _string(cells, name: str) -> str:
    value = cells.get(name)
    if value is None or value == "":
        raise ValueError(f"motion_v1 missing nonempty {name!r}")
    return str(value)


def _optional_string(value) -> str | None:
    return None if value is None or value == "" else str(value)


def _strings(value) -> list[str]:
    return [str(item) for item in (value or [])]


def _asset_paths(cells, name: str, expected_count: int) -> list[str]:
    paths = _strings(cells.get(name))
    if len(paths) != expected_count or any(not path for path in paths):
        raise ValueError(f"motion_v1 {name!r} must contain {expected_count} nonempty paths, got {len(paths)}")
    return paths


def _positive_vector(cells, name: str, expected_count: int) -> np.ndarray:
    values = np.asarray(cells.get(name) or [], dtype=np.float64)
    if values.shape != (expected_count,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"motion_v1 {name!r} must have shape ({expected_count},) with positive finite values")
    return values


def _unique_names(cells, name: str, *, require_nonempty: bool) -> list[str]:
    names = _strings(cells.get(name))
    if require_nonempty and not names:
        raise ValueError(f"motion_v1 missing nonempty {name!r}")
    if len(names) != len(set(names)):
        raise ValueError(f"motion_v1 {name!r} must contain unique names")
    return names


def _positive_float(cells, name: str) -> float:
    value = float(cells.get(name, np.nan))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"motion_v1 {name!r} must be finite and positive, got {value}")
    return value


def _finite_array(cells, name: str, *, ndim: int) -> np.ndarray:
    if name not in cells or cells[name] is None:
        raise ValueError(f"motion_v1 missing {name!r}")
    values = np.asarray(cells[name])
    if values.ndim != ndim:
        raise ValueError(f"motion_v1 {name!r} must have rank {ndim}, got shape {values.shape}")
    if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError(f"motion_v1 {name!r} must contain only finite numeric values")
    return np.ascontiguousarray(values)


def _trajectory(cells, name: str, num_frames: int, trailing_shape: tuple[int, ...]) -> np.ndarray:
    values = _finite_array(cells, name, ndim=1 + len(trailing_shape))
    expected = (num_frames,) + trailing_shape
    if values.shape != expected:
        raise ValueError(f"motion_v1 {name!r} must have shape {expected}, got {values.shape}")
    return values


def _quaternion_trajectory(cells, name: str, num_frames: int, leading_shape: tuple[int, ...]) -> np.ndarray:
    values = _trajectory(cells, name, num_frames, leading_shape + (4,))
    return _validated_quaternions(values, name)


def _validated_quaternions(values: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= np.finfo(values.dtype).eps):
        raise ValueError(f"motion_v1 {name!r} contains a zero quaternion")
    if not np.allclose(norms, 1.0, atol=1.0e-4, rtol=0.0):
        max_error = float(np.max(np.abs(norms - 1.0)))
        raise ValueError(f"motion_v1 {name!r} quaternions are not unit length; max error {max_error:.3e}")
    return np.ascontiguousarray(values / norms)


def _articulation(value, num_frames: int) -> np.ndarray:
    if value is None or len(value) == 0:
        return np.zeros((num_frames, 0), dtype=np.float32)
    values = np.asarray(value)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != num_frames or not np.all(np.isfinite(values)):
        raise ValueError(
            f"motion_v1 'object_articulation' must be finite with shape ({num_frames}, A), got {values.shape}"
        )
    return np.ascontiguousarray(values)


def _per_side_values(cells, name: str, sides: tuple[str, ...]) -> dict[str, object]:
    values = cells.get(name)
    if not isinstance(values, (list, tuple)) or len(values) != len(sides):
        actual = len(values) if isinstance(values, (list, tuple)) else 0
        raise ValueError(f"motion_v1 {name!r} must have one entry per hand side; expected {len(sides)}, got {actual}")
    return dict(zip(sides, values, strict=True))


def _contacts(cells, sides: tuple[str, ...], num_frames: int, num_object_bodies: int):
    positions = cells.get("hand_object_contact_positions")
    normals = cells.get("hand_object_contact_normals")
    part_ids = cells.get("hand_object_contact_part_ids")
    empty_payloads = tuple(
        _empty_per_side_contact_payload(value, len(sides)) for value in (positions, normals, part_ids)
    )
    if all(empty_payloads):
        empty_vec = {side: np.zeros((num_frames, 0, 3), dtype=np.float32) for side in sides}
        empty_ids = {side: np.zeros((num_frames, 0), dtype=np.int32) for side in sides}
        return empty_vec, dict(empty_vec), empty_ids
    if any(empty_payloads):
        raise ValueError("motion_v1 contact positions, normals, and part IDs must be provided together")

    position_values = _per_side_values(cells, "hand_object_contact_positions", sides)
    normal_values = _per_side_values(cells, "hand_object_contact_normals", sides)
    part_id_values = _per_side_values(cells, "hand_object_contact_part_ids", sides)
    positions_by_side: dict[str, np.ndarray] = {}
    normals_by_side: dict[str, np.ndarray] = {}
    part_ids_by_side: dict[str, np.ndarray] = {}
    for side in sides:
        if not position_values[side] and not normal_values[side] and not part_id_values[side]:
            positions_by_side[side] = np.zeros((num_frames, 0, 3), dtype=np.float32)
            normals_by_side[side] = np.zeros((num_frames, 0, 3), dtype=np.float32)
            part_ids_by_side[side] = np.zeros((num_frames, 0), dtype=np.int32)
            continue
        positions_for_side = np.asarray(position_values[side])
        normals_for_side = np.asarray(normal_values[side])
        ids_for_side = np.asarray(part_id_values[side])
        if (
            positions_for_side.ndim != 3
            or positions_for_side.shape[0] != num_frames
            or positions_for_side.shape[-1] != 3
        ):
            raise ValueError(f"invalid hand_object_contact_positions[{side}] shape {positions_for_side.shape}")
        if normals_for_side.shape != positions_for_side.shape:
            raise ValueError(
                f"contact normals[{side}] shape {normals_for_side.shape} "
                f"does not match positions {positions_for_side.shape}"
            )
        if ids_for_side.shape != positions_for_side.shape[:-1]:
            raise ValueError(
                f"contact part IDs[{side}] shape {ids_for_side.shape} "
                f"does not match positions {positions_for_side.shape}"
            )
        if not np.all(np.isfinite(positions_for_side)) or not np.all(np.isfinite(normals_for_side)):
            raise ValueError(f"motion_v1 {side} contact positions and normals must be finite")
        if not np.issubdtype(ids_for_side.dtype, np.number) or not np.all(np.isfinite(ids_for_side)):
            raise ValueError(f"motion_v1 {side} contact part IDs must be finite numeric values")
        if np.any(ids_for_side != np.floor(ids_for_side)):
            raise ValueError(f"motion_v1 {side} contact part IDs must be integers")
        ids_for_side = ids_for_side.astype(np.int32)
        if np.any(ids_for_side < 0) or np.any(ids_for_side > num_object_bodies):
            raise ValueError(f"motion_v1 {side} contact part IDs must be in [0, {num_object_bodies}]")
        active_normal_norm = np.linalg.norm(normals_for_side, axis=-1)[ids_for_side > 0]
        if active_normal_norm.size and not np.allclose(active_normal_norm, 1.0, atol=1.0e-4, rtol=0.0):
            raise ValueError(f"motion_v1 {side} active contact normals must be unit length")
        positions_by_side[side] = np.ascontiguousarray(positions_for_side)
        normals_by_side[side] = np.ascontiguousarray(normals_for_side)
        part_ids_by_side[side] = np.ascontiguousarray(ids_for_side)
    return positions_by_side, normals_by_side, part_ids_by_side


def _empty_per_side_contact_payload(value, num_sides: int) -> bool:
    if value is None:
        return True
    try:
        return len(value) == num_sides and all(side is None or len(side) == 0 for side in value)
    except TypeError:
        return False


def _empty_per_side_payload(value, num_sides: int) -> bool:
    if value is None:
        return True
    try:
        return len(value) == 0 or (len(value) == num_sides and all(side is None or len(side) == 0 for side in value))
    except TypeError:
        return False


def _contact_activity(
    value,
    sides: tuple[str, ...],
    num_frames: int,
    contact_part_ids: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate per-hand binary labels, deriving them from contact IDs when absent."""
    if value is None or _empty_per_side_contact_payload(value, len(sides)):
        return {
            side: np.ascontiguousarray(np.any(contact_part_ids[side] > 0, axis=1).astype(np.float32)) for side in sides
        }
    values = np.asarray(value)
    expected = (len(sides), num_frames)
    if values.shape != expected or not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
        raise ValueError(f"motion_v1 'hand_contact_active' must be finite numeric values with shape {expected}")
    if np.any((values != 0.0) & (values != 1.0)):
        raise ValueError("motion_v1 'hand_contact_active' must contain only binary 0/1 values")
    return {side: np.ascontiguousarray(values[index], dtype=np.float32) for index, side in enumerate(sides)}
