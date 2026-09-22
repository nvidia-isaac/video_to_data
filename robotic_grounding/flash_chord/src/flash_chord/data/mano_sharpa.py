# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a retargeted ManoSharpa parquet into a :class:`~flash_chord.data.reference.Reference`.

The parquet is single-row with each cell a nested per-frame array. Robot wrist/finger
fields are populated only in retargeted (``_processed``) parquets; ``_loaded`` parquets
carry MANO + object + contact but leave the robot fields empty. Quaternions are wxyz.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from flash_chord.assets.registry import resolve_object_asset_specs
from flash_chord.data.parquet import read_parquet_row
from flash_chord.data.reference import ObjectAssetSpec, ReferenceMetadata, reference_frame_slice
from flash_chord.data.resampling import (
    lerp as _lerp,
    nearest as _nearest,
    playback_times as _playback_times,
    slerp_tracks_wxyz,
    slerp_wxyz as _slerp_wxyz,
)

_SIDES = ("left", "right")


@dataclass
class ManoSharpaReference:
    """In-memory ManoSharpa reference (host numpy arrays), implementing ``Reference``."""

    _metadata: ReferenceMetadata
    _fps: float
    _wrist_pos_w: dict[str, np.ndarray]  # side -> (T, 3)
    _wrist_quat_w: dict[str, np.ndarray]  # side -> (T, 4) wxyz
    _finger_joint_pos: dict[str, np.ndarray]  # side -> (T, Nf)
    _finger_joint_names: dict[str, list[str]]
    _object_body_pos_w: np.ndarray  # (T, B, 3)
    _object_body_quat_w: np.ndarray  # (T, B, 4) wxyz
    _object_articulation: np.ndarray  # (T, A)
    _contact_pos_w: dict[str, np.ndarray]  # side -> (T, K, 3)
    _contact_normal_w: dict[str, np.ndarray]  # side -> (T, K, 3)
    _contact_part_ids: dict[str, np.ndarray]  # side -> (T, K)
    _object_name: str
    _object_body_names: list[str]
    _object_mesh_paths: list[str]
    _object_urdf_paths: list[str]
    _object_mesh_radius: np.ndarray  # (B,) per-body bounding-ball radius (wrench-support torque scale rc)
    _object_assets: tuple[ObjectAssetSpec, ...]
    _frame_pos_w: dict[str, np.ndarray]  # side -> (T, F, 3) robot task-frame positions
    _frame_names: dict[str, list[str]]  # side -> F robot task-frame names

    @property
    def num_frames(self) -> int:
        return self._object_body_pos_w.shape[0]

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def metadata(self) -> ReferenceMetadata:
        return self._metadata

    @property
    def sides(self) -> tuple[str, ...]:
        return _SIDES

    def wrist_pos_w(self, side: str) -> np.ndarray:
        return self._wrist_pos_w[side]

    def wrist_quat_w(self, side: str) -> np.ndarray:
        return self._wrist_quat_w[side]

    def finger_joint_pos(self, side: str) -> np.ndarray:
        return self._finger_joint_pos[side]

    def finger_joint_names(self, side: str) -> list[str]:
        return self._finger_joint_names[side]

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
        return np.ascontiguousarray(np.any(self._contact_part_ids[side] > 0, axis=1).astype(np.float32))

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

    def frame_pos_w(self, side: str) -> np.ndarray:
        return self._frame_pos_w[side]

    def frame_names(self, side: str) -> list[str]:
        return self._frame_names[side]

    def frame_window(self, start_frame: int = 0, end_frame: int = -1) -> ManoSharpaReference:
        """Return a copy restricted to the half-open source-frame window."""
        frames = reference_frame_slice(self.num_frames, start_frame, end_frame)
        if frames.start == 0 and frames.stop == self.num_frames:
            return self

        def sliced(values: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(values[frames])

        return replace(
            self,
            _wrist_pos_w={side: sliced(values) for side, values in self._wrist_pos_w.items()},
            _wrist_quat_w={side: sliced(values) for side, values in self._wrist_quat_w.items()},
            _finger_joint_pos={side: sliced(values) for side, values in self._finger_joint_pos.items()},
            _object_body_pos_w=sliced(self._object_body_pos_w),
            _object_body_quat_w=sliced(self._object_body_quat_w),
            _object_articulation=sliced(self._object_articulation),
            _contact_pos_w={side: sliced(values) for side, values in self._contact_pos_w.items()},
            _contact_normal_w={side: sliced(values) for side, values in self._contact_normal_w.items()},
            _contact_part_ids={side: sliced(values) for side, values in self._contact_part_ids.items()},
            _frame_pos_w={side: sliced(values) for side, values in self._frame_pos_w.items()},
        )


def _arr(cells: dict, name: str) -> np.ndarray | None:
    if name not in cells:
        return None
    a = np.asarray(cells[name])
    return None if a.size == 0 else a


def load_mano_sharpa(
    parquet_path: str,
    control_fps: float | None = None,
    motion_speed: float = 1.0,
) -> ManoSharpaReference:
    """Load a ManoSharpa parquet and optionally resample it for control-rate playback.

    ``motion_speed`` scales source-time advancement per control step. Positions and joints use linear
    interpolation, quaternions use SLERP, and contact fields use nearest-neighbor sampling.
    """
    if control_fps is None and motion_speed != 1.0:
        raise ValueError("control_fps is required when motion_speed is not 1.0")
    row = read_parquet_row(parquet_path)
    cells = dict(row.cells)
    src_fps = float(cells["fps"])

    obj_pos = _arr(cells, "object_body_position")  # (T, B, 3)
    obj_quat = _arr(cells, "object_body_wxyz")  # (T, B, 4)
    if obj_pos is None or obj_quat is None:
        raise ValueError("parquet missing object_body_position / object_body_wxyz")
    if obj_pos.ndim != 3 or obj_pos.shape[-1] != 3 or not np.all(np.isfinite(obj_pos)):
        raise ValueError(f"object_body_position must be finite with shape (T, B, 3), got {obj_pos.shape}")
    if obj_quat.shape != obj_pos.shape[:-1] + (4,) or not np.all(np.isfinite(obj_quat)):
        raise ValueError(f"object_body_wxyz must be finite with shape (T, B, 4), got {obj_quat.shape}")
    quaternion_norm = np.linalg.norm(obj_quat, axis=-1, keepdims=True)
    if np.any(quaternion_norm <= np.finfo(obj_quat.dtype).eps):
        raise ValueError("object_body_wxyz contains a zero quaternion")
    if not np.allclose(quaternion_norm, 1.0, atol=1.0e-4, rtol=0.0):
        raise ValueError("object_body_wxyz quaternions must be unit length")
    obj_quat = np.ascontiguousarray(obj_quat / quaternion_norm)
    num_frames = obj_pos.shape[0]
    num_object_bodies = obj_pos.shape[1]
    object_name = str(cells.get("object_name") or "")
    object_body_names = [str(value) for value in (cells.get("object_body_names") or [])]
    if (
        not object_name
        or len(object_body_names) != num_object_bodies
        or len(set(object_body_names)) != num_object_bodies
        or any(not name for name in object_body_names)
    ):
        raise ValueError("object names must be nonempty, unique, and match the reference body axis")
    object_mesh_paths = [str(value) for value in (cells.get("object_mesh_paths") or [])]
    if len(object_mesh_paths) != num_object_bodies or any(not path for path in object_mesh_paths):
        raise ValueError("object mesh paths must be nonempty and match the reference body axis")
    object_urdf_paths = [str(value) for value in (cells.get("object_urdf_paths") or [])]
    object_mesh_radius = np.asarray(cells.get("object_mesh_radius") or [], dtype=np.float64)
    if (
        object_mesh_radius.shape != (num_object_bodies,)
        or not np.all(np.isfinite(object_mesh_radius))
        or np.any(object_mesh_radius <= 0.0)
    ):
        raise ValueError("object mesh radii must be positive, finite, and match the reference body axis")
    art = _arr(cells, "object_articulation")
    art = np.zeros((num_frames, 0)) if art is None else art.reshape(num_frames, -1)
    if not np.all(np.isfinite(art)):
        raise ValueError("object articulation trajectory must be finite")
    if object_urdf_paths and art.shape[1]:
        if art.shape[1] != 1 or not np.all(art == 0.0):
            raise ValueError(
                "ManoSharpaData references with independent rigid-object URDFs may contain only one "
                "all-zero articulation placeholder column"
            )
        art = np.zeros((num_frames, 0), dtype=art.dtype)
    source_dataset = _optional_string(cells.get("source_dataset")) or _dataset_from_path(row.path)
    object_assets = resolve_object_asset_specs(
        source_dataset=source_dataset,
        object_name=object_name,
        body_names=object_body_names,
        mesh_paths=object_mesh_paths,
        urdf_paths=object_urdf_paths,
        num_articulations=art.shape[1],
        reference_path=row.path,
    )

    t_src, t_tgt = (
        _playback_times(num_frames, src_fps, control_fps, motion_speed) if control_fps is not None else (None, None)
    )
    resample = control_fps is not None and (control_fps != src_fps or motion_speed != 1.0)

    def lerp(a):
        return _lerp(a, t_src, t_tgt) if resample and a is not None else a

    def slerp_bodies(q):  # q: (T, B, 4)
        if not resample or q is None:
            return q
        return slerp_tracks_wxyz(q, t_src, t_tgt)

    def near(a):
        return _nearest(a, t_src, t_tgt) if resample and a is not None else a

    wrist_pos, wrist_quat, finger, finger_names, c_pos, c_norm, c_pid = {}, {}, {}, {}, {}, {}, {}
    frame_pos, frame_names = {}, {}
    for side in _SIDES:
        wp_ = _arr(cells, f"robot_{side}_wrist_position")
        wq_ = _arr(cells, f"robot_{side}_wrist_wxyz")
        fj = _arr(cells, f"robot_{side}_finger_joints")
        if wp_ is None or wp_.shape != (num_frames, 3) or not np.all(np.isfinite(wp_)):
            raise ValueError(f"ManoSharpaData {side} wrist position must be finite with shape ({num_frames}, 3)")
        if wq_ is None or wq_.shape != (num_frames, 4) or not np.all(np.isfinite(wq_)):
            raise ValueError(f"ManoSharpaData {side} wrist quaternion must be finite with shape ({num_frames}, 4)")
        wrist_norm = np.linalg.norm(wq_, axis=-1, keepdims=True)
        if np.any(wrist_norm <= np.finfo(wq_.dtype).eps) or not np.allclose(
            wrist_norm,
            1.0,
            atol=1.0e-4,
            rtol=0.0,
        ):
            raise ValueError(f"ManoSharpaData {side} wrist quaternions must be unit length")
        wq_ = np.ascontiguousarray(wq_ / wrist_norm)
        if fj is None or fj.ndim != 2 or fj.shape[0] != num_frames or not np.all(np.isfinite(fj)):
            raise ValueError(f"ManoSharpaData {side} finger joints must be finite with shape ({num_frames}, N)")
        names = [str(x) for x in (cells.get(f"{side}_robot_finger_joint_names") or [])]
        if len(names) != fj.shape[1] or len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError(f"ManoSharpaData {side} finger joint names must be unique and match the joint axis")
        wrist_pos[side] = lerp(wp_)
        wrist_quat[side] = _slerp_wxyz(wq_, t_src, t_tgt) if resample else wq_
        finger[side] = lerp(fj)
        finger_names[side] = names
        cp = _arr(cells, f"mano_{side}_object_contact_positions")
        cn = _arr(cells, f"mano_{side}_object_contact_normals")
        cid = _arr(cells, f"mano_{side}_object_contact_part_ids")
        if cp is None or cn is None or cid is None:
            raise ValueError(f"ManoSharpaData {side} contact positions, normals, and part IDs are required together")
        if cp.ndim != 3 or cp.shape[0] != num_frames or cp.shape[-1] != 3 or cn.shape != cp.shape:
            raise ValueError(f"ManoSharpaData {side} contacts must have matching (T, K, 3) position/normal shapes")
        if cid.shape != cp.shape[:-1]:
            raise ValueError(f"ManoSharpaData {side} contact part IDs must match the contact slot axes")
        if not np.all(np.isfinite(cp)) or not np.all(np.isfinite(cn)) or not np.all(np.isfinite(cid)):
            raise ValueError(f"ManoSharpaData {side} contact fields must be finite")
        if np.any(cid != np.floor(cid)) or np.any(cid < 0) or np.any(cid > num_object_bodies):
            raise ValueError(f"ManoSharpaData {side} contact part IDs must be integers in [0, {num_object_bodies}]")
        cid = cid.astype(np.int32)
        active_normal_norm = np.linalg.norm(cn, axis=-1)[cid > 0]
        if active_normal_norm.size and not np.allclose(active_normal_norm, 1.0, atol=1.0e-4, rtol=0.0):
            raise ValueError(f"ManoSharpaData {side} active contact normals must be unit length")
        # contacts use NEAREST (not lerp): a slot's position/normal must not be interpolated across an
        # active->inactive transition, which would place it at a garbage midpoint flying off the object.
        c_pos[side] = near(cp) if cp is not None else np.zeros((num_frames, 0, 3))
        c_norm[side] = near(cn) if cn is not None else np.zeros((num_frames, 0, 3))
        c_pid[side] = near(cid) if cid is not None else np.zeros((num_frames, 0), dtype=int)
        fr = _arr(cells, f"robot_{side}_frames")  # (T, F, 7) pos + wxyz per robot task frame
        names = [str(x) for x in (cells.get(f"{side}_robot_frame_names") or [])]
        if fr is None or fr.ndim != 3 or fr.shape[0] != num_frames or fr.shape[-1] != 7:
            raise ValueError(f"ManoSharpaData {side} robot frames must have shape ({num_frames}, F, 7)")
        if len(names) != fr.shape[1] or len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError(f"ManoSharpaData {side} robot frame names must be unique and match the frame axis")
        if not np.all(np.isfinite(fr)):
            raise ValueError(f"ManoSharpaData {side} robot frames must be finite")
        frame_pos[side] = lerp(np.ascontiguousarray(fr[..., :3]))
        frame_names[side] = names

    if not resample:
        art_out = art
    elif art.shape[1]:
        art_out = lerp(art)
    else:
        assert t_tgt is not None
        art_out = np.zeros((len(t_tgt), 0))

    return ManoSharpaReference(
        _metadata=ReferenceMetadata(
            source_path=str(row.path),
            source_fps=src_fps,
            is_resampled=resample,
            schema_version=_optional_string(cells.get("schema_version")),
            motion_kind=_optional_string(cells.get("motion_kind")),
            source_dataset=source_dataset,
            sequence_id=_optional_string(cells.get("sequence_id")),
            robot_name=_optional_string(cells.get("robot_name")),
            raw_motion_file=_optional_string(cells.get("raw_motion_file")),
            coord_frame=_optional_string(cells.get("coord_frame")),
        ),
        _object_name=object_name,
        _fps=(control_fps if control_fps is not None else src_fps),
        _wrist_pos_w=wrist_pos,
        _wrist_quat_w=wrist_quat,
        _finger_joint_pos=finger,
        _finger_joint_names=finger_names,
        _object_body_pos_w=lerp(obj_pos),
        _object_body_quat_w=slerp_bodies(obj_quat),
        _object_articulation=art_out,
        _contact_pos_w=c_pos,
        _contact_normal_w=c_norm,
        _contact_part_ids=c_pid,
        _object_body_names=object_body_names,
        _object_mesh_paths=object_mesh_paths,
        _object_urdf_paths=object_urdf_paths,
        _object_mesh_radius=object_mesh_radius,
        _object_assets=object_assets,
        _frame_pos_w=frame_pos,
        _frame_names=frame_names,
    )


def _optional_string(value) -> str | None:
    return None if value is None or value == "" else str(value)


def _dataset_from_path(path: Path) -> str | None:
    parts = path.parts
    try:
        index = parts.index("human_motion_data")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None
