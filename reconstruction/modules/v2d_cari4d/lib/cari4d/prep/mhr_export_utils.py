from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np

from prep.mhr_depth_h5 import SOURCE_LOCK_SUFFIX, _register_h5py_fork_child_reset, _shared_source_lock, clear_depth_h5_cache, read_metric_depth
from prep.mhr_effective_masks import EffectiveMask, MASK_SOURCE_RAW, MASK_SOURCE_RAW_EMPTY_INVALID_POSE, effective_mask_frame_usability, effective_mask_kind_identity, path_identity, read_effective_mask_from_handle, validate_effective_mask_handle
from prep.mhr_ffv1_sidecar import clear_ffv1_cache
from prep.mhr_rgb_h5 import read_rgb_h5_frame, rgb_h5_frame_count
from prep.mhr_sensor_depth_h5 import read_sensor_depth_h5_frame


MHR_CAMERA_NAMES = (
    "front_stereo_camera_left",
    "back_stereo_camera_left",
    "right_stereo_camera_left",
    "left_stereo_camera_left",
)
MHR_EDEX_LEFT_CAMERA_IDXS = (0, 2, 6, 4)
DEPTH_IMAGE_SCALE = 65535.0
H5_CACHE_MAX_HANDLES = 32
EFFECTIVE_MASK_VALIDATION_CACHE_MAX_ENTRIES = 128
CANONICAL_OBJECT_MASK_LOADER_REVISION = "required-effective-sidecar-invalid-pose-empty-error-v1"


class UnusableObjectMaskError(ValueError):
    pass


@dataclass
class _H5CacheEntry:
    identity: tuple[int, int, int, int]
    handle: h5py.File
    owner_pid: int
    pins: int = 0


_H5_HANDLES: OrderedDict[str, _H5CacheEntry] = OrderedDict()
_H5_CACHE_PID = os.getpid()
_H5_CACHE_CONDITION = threading.Condition(threading.RLock())
_VALIDATED_EFFECTIVE_MASKS: OrderedDict[tuple[Any, ...], None] = OrderedDict()
_EFFECTIVE_MASK_VALIDATION_LOCK = threading.RLock()


def _after_h5_cache_fork_child() -> None:
    global _H5_HANDLES, _H5_CACHE_PID, _H5_CACHE_CONDITION, _VALIDATED_EFFECTIVE_MASKS, _EFFECTIVE_MASK_VALIDATION_LOCK
    for entry in _H5_HANDLES.values():
        if entry.handle.id.valid:
            entry.handle.close()
    _H5_HANDLES = OrderedDict()
    _H5_CACHE_PID = os.getpid()
    _H5_CACHE_CONDITION = threading.Condition(threading.RLock())
    _VALIDATED_EFFECTIVE_MASKS = OrderedDict()
    _EFFECTIVE_MASK_VALIDATION_LOCK = threading.RLock()


_register_h5py_fork_child_reset("mhr-export-utils", _after_h5_cache_fork_child)


def _file_identity(path: str | Path) -> tuple[int, int, int, int]:
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _open_stable_h5(path: str) -> tuple[tuple[int, int, int, int], h5py.File]:
    for _ in range(3):
        before = _file_identity(path)
        handle = h5py.File(path, "r")
        after = _file_identity(path)
        if before == after:
            return after, handle
        handle.close()
    raise RuntimeError(f"HDF5 path changed repeatedly while opening: {path}")


def _close_h5_entry(entry: _H5CacheEntry) -> None:
    if entry.handle.id.valid:
        entry.handle.close()


def _reset_h5_cache_pid_locked() -> None:
    global _H5_CACHE_PID
    pid = os.getpid()
    if _H5_CACHE_PID == pid:
        return
    for entry in list(_H5_HANDLES.values()):
        _close_h5_entry(entry)
    _H5_HANDLES.clear()
    _H5_CACHE_PID = pid


def _evict_unpinned_h5_locked() -> bool:
    for path_key, entry in list(_H5_HANDLES.items()):
        if entry.pins == 0:
            del _H5_HANDLES[path_key]
            _close_h5_entry(entry)
            return True
    return False


@contextmanager
def _borrow_cached_h5(path: str | Path) -> Iterator[h5py.File]:
    path_key = str(Path(path).resolve())
    entry = None
    with _H5_CACHE_CONDITION:
        while entry is None:
            _reset_h5_cache_pid_locked()
            identity = _file_identity(path_key)
            cached = _H5_HANDLES.get(path_key)
            if cached is not None and cached.identity == identity and cached.handle.id.valid:
                cached.pins += 1
                _H5_HANDLES.move_to_end(path_key)
                entry = cached
                break
            if cached is not None:
                if cached.pins:
                    _H5_CACHE_CONDITION.wait()
                    continue
                del _H5_HANDLES[path_key]
                _close_h5_entry(cached)
            retry = False
            while len(_H5_HANDLES) >= H5_CACHE_MAX_HANDLES:
                if _evict_unpinned_h5_locked():
                    continue
                _H5_CACHE_CONDITION.wait()
                retry = True
                break
            if retry:
                continue
            identity, handle = _open_stable_h5(path_key)
            entry = _H5CacheEntry(identity=identity, handle=handle, owner_pid=os.getpid(), pins=1)
            _H5_HANDLES[path_key] = entry
    try:
        yield entry.handle
    finally:
        with _H5_CACHE_CONDITION:
            entry.pins -= 1
            _H5_CACHE_CONDITION.notify_all()


@contextmanager
def _borrow_h5(path: str | Path) -> Iterator[h5py.File]:
    with _shared_source_lock(path):
        with _borrow_cached_h5(path) as handle:
            yield handle


def clear_h5_cache() -> None:
    with _H5_CACHE_CONDITION:
        _reset_h5_cache_pid_locked()
        while any(entry.pins for entry in _H5_HANDLES.values()):
            _H5_CACHE_CONDITION.wait()
        for entry in list(_H5_HANDLES.values()):
            _close_h5_entry(entry)
        _H5_HANDLES.clear()
        _H5_CACHE_CONDITION.notify_all()
    with _EFFECTIVE_MASK_VALIDATION_LOCK:
        _VALIDATED_EFFECTIVE_MASKS.clear()
    clear_depth_h5_cache()
    clear_ffv1_cache()


def _source_h5_path(export_seq: str | Path, dirname: str, camera_id: int) -> Path:
    return Path(export_seq) / dirname / f"{MHR_CAMERA_NAMES[camera_id]}.h5"


def effective_mask_root(export_seq: str | Path) -> Path:
    configured = os.environ.get("MHR_EFFECTIVE_MASK_ROOT", "").strip()
    if configured:
        return Path(configured)
    export_seq = Path(export_seq)
    if export_seq.parent.name == "data_export":
        return export_seq.parent.parent / "effective_masks"
    return export_seq / "effective_masks"


def effective_mask_path(export_seq: str | Path, camera_id: int) -> Path:
    export_seq = Path(export_seq)
    return effective_mask_root(export_seq) / export_seq.name / f"{MHR_CAMERA_NAMES[camera_id]}.h5"


def effective_mask_sidecar_required(export_seq: str | Path) -> bool:
    return bool(os.environ.get("MHR_EFFECTIVE_MASK_ROOT", "").strip()) or effective_mask_root(export_seq).exists()


def raw_mask_source_path(export_seq: str | Path, kind: str, camera_id: int) -> Path:
    if kind not in {"human", "object"}:
        raise ValueError(f"Unsupported mask kind {kind}")
    dirname = "human_masks" if kind == "human" else "object_masks"
    h5_path = _source_h5_path(export_seq, dirname, camera_id)
    if h5_path.is_file():
        return h5_path
    frame_root = Path(export_seq) / dirname / MHR_CAMERA_NAMES[camera_id]
    if not frame_root.is_dir():
        raise FileNotFoundError(f"No raw {kind} mask source found for camera {camera_id} under {export_seq}")
    return frame_root


def raw_mask_identities(export_seq: str | Path, camera_id: int) -> dict[str, dict[str, Any]]:
    identities = {}
    for kind in ("human", "object"):
        source = raw_mask_source_path(export_seq, kind, camera_id)
        if not source.is_file():
            raise ValueError(f"Effective-mask sidecars require canonical H5 mask sources, got {source}")
        identities[kind] = path_identity(source)
    return identities


def pose_validity_source_identity(export_seq: str | Path) -> dict[str, dict[str, Any] | None]:
    export_seq = Path(export_seq)
    return {name: path_identity(export_seq / name) if (export_seq / name).is_file() else None for name in ("pose_valid_mask.npy", "failure_segments.json", "interaction_trim.json")}


def _rgb_timeline_cache_identity(export_seq: str | Path, camera_id: int) -> tuple[Any, ...]:
    export_seq = Path(export_seq)
    h5_path = _source_h5_path(export_seq, "images", camera_id)
    if h5_path.is_file():
        return "h5", str(h5_path.resolve()), _file_identity(h5_path)
    frame_root = export_seq / "images" / MHR_CAMERA_NAMES[camera_id]
    if not frame_root.is_dir():
        raise FileNotFoundError(f"No PNG or H5 RGB frames found for camera {camera_id} under {export_seq}")
    stat = frame_root.stat()
    return "png", str(frame_root.resolve()), stat.st_dev, stat.st_ino, stat.st_mtime_ns


def _resolve_effective_mask_sidecar(export_seq: str | Path, camera_id: int) -> Path | None:
    sidecar = effective_mask_path(export_seq, camera_id)
    if sidecar.is_file():
        return sidecar
    if effective_mask_sidecar_required(export_seq):
        raise FileNotFoundError(f"Required effective-mask sidecar is missing for {Path(export_seq).name}/{MHR_CAMERA_NAMES[camera_id]}: {sidecar}")
    return None


def mask_input_identity(export_seq: str | Path, kind: str, camera_id: int) -> dict[str, Any]:
    raw_path = raw_mask_source_path(export_seq, kind, camera_id)
    if raw_path.is_file():
        raw_identity: dict[str, Any] = {"storage": "h5", "source": path_identity(raw_path)}
    else:
        entries = sorted(raw_path.glob("*.png"))
        raw_identity = {"storage": "png", "root": str(raw_path.resolve()), "frame_count": len(entries), "total_bytes": int(sum(path.stat().st_size for path in entries)), "latest_mtime_ns": int(max((path.stat().st_mtime_ns for path in entries), default=0))}
    sidecar = _resolve_effective_mask_sidecar(export_seq, camera_id)
    if sidecar is not None:
        with _borrow_h5(sidecar) as handle:
            _validate_effective_mask_sidecar(export_seq, camera_id, sidecar, handle)
    identity = {"raw": raw_identity, "effective": effective_mask_kind_identity(sidecar, kind) if sidecar is not None else None}
    if kind == "object":
        identity["loader_revision"] = CANONICAL_OBJECT_MASK_LOADER_REVISION
    return identity


def _frame_index(frame_name: str) -> int:
    if not str(frame_name).isdigit():
        raise ValueError(f"H5-backed Daniel HOI frame names must be numeric, got {frame_name}")
    return int(frame_name)


def resolve_object_mesh_path(export_seq: str | Path) -> Path:
    export_seq = Path(export_seq)
    candidates = (export_seq / "object_mesh" / "output_aligned.glb", export_seq / "object_template" / "output.glb")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No object mesh found under {export_seq}; checked {', '.join(str(path) for path in candidates)}")


def load_edex(export_seq: str | Path) -> list[Mapping[str, Any]]:
    with (Path(export_seq) / "edex").open("r") as f:
        return json.load(f)


def camera_calibration(edex: list[Mapping[str, Any]], camera_id: int) -> tuple[np.ndarray, np.ndarray]:
    cam = edex[0]["cameras"][MHR_EDEX_LEFT_CAMERA_IDXS[camera_id]]
    intr = cam["intrinsics"]
    focal = np.asarray(intr["focal"], dtype=np.float32)
    principal = np.asarray(intr["principal"], dtype=np.float32)
    K = np.array(
        [[focal[0], 0.0, principal[0]], [0.0, focal[1], principal[1]], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :4] = np.asarray(cam["transform"], dtype=np.float32)
    w2c = np.linalg.inv(c2w).astype(np.float32)
    return K, w2c


def _is_raw_export_depth_root(export_seq: Path, depth_root: str | Path | None) -> bool:
    if depth_root is None:
        return True
    try:
        return Path(depth_root).resolve() == (export_seq / "depth").resolve()
    except FileNotFoundError:
        return Path(depth_root) == export_seq / "depth"


def _camera_center_from_w2c(w2c: np.ndarray) -> np.ndarray:
    return (-w2c[:3, :3].T @ w2c[:3, 3]).astype(np.float32)


def depthimage_png_to_depth_m(depth_png: np.ndarray) -> np.ndarray:
    """Decode the video_to_data DepthImage inverse-depth PNG format.

    This follows ``v2d.common.datatypes.DepthImage.from_pil_image``:
    ``depth_m = 1 / (pixel / 65535) - 1``.  The flat export demo loads
    ``depth/<camera>/<frame>.png`` with ``DepthImage.load()``, so the exported
    sequence depth folder must use this decoder rather than a disparity formula.
    """

    inverse_depth = np.asarray(depth_png, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1.0 / (inverse_depth / DEPTH_IMAGE_SCALE) - 1.0).astype(np.float32)


def interaction_trim_range(export_seq: str | Path, source_frame_count: int) -> tuple[int, int]:
    export_seq = Path(export_seq)
    source_frame_count = int(source_frame_count)
    trim_path = export_seq / "interaction_trim.json"
    if not trim_path.is_file():
        return 0, source_frame_count
    trim = json.loads(trim_path.read_text())
    start = int(trim["export_source_start_frame"])
    end = int(trim["export_source_end_frame"])
    export_frame_count = int(trim["export_frame_count"])
    declared_source_count = int(trim.get("source_frame_count", source_frame_count))
    if declared_source_count != source_frame_count:
        raise ValueError(f"interaction_trim.json source frame count differs from RGB timeline for {export_seq}: {declared_source_count} != {source_frame_count}")
    if start < 0 or end < start or end > source_frame_count or end - start != export_frame_count:
        raise ValueError(f"Invalid interaction trim [{start},{end})/{export_frame_count} for {source_frame_count} source frames: {export_seq}")
    return start, end


def source_frame_indices(export_seq: str | Path, source_frame_count: int) -> np.ndarray:
    start, end = interaction_trim_range(export_seq, source_frame_count)
    return np.arange(start, end, dtype=np.int64)


def frame_names(export_seq: str | Path, camera_id: int = 0) -> list[str]:
    export_seq = Path(export_seq)
    root = export_seq / "images" / MHR_CAMERA_NAMES[camera_id]
    png_names = [path.stem for path in sorted(root.glob("*.png"))]
    if png_names:
        start, end = interaction_trim_range(export_seq, len(png_names))
        selected = png_names[start:end]
        expected = [f"{index:06d}" for index in range(start, end)]
        if selected != expected:
            raise ValueError(f"PNG RGB timeline does not match source-frame identities for {export_seq}/{MHR_CAMERA_NAMES[camera_id]}")
        return selected
    h5_path = _source_h5_path(export_seq, "images", camera_id)
    if not h5_path.is_file():
        raise FileNotFoundError(f"No PNG or H5 RGB frames found for camera {camera_id} under {export_seq}")
    with _borrow_h5(h5_path) as handle:
        source_count = rgb_h5_frame_count(handle)
    start, end = interaction_trim_range(export_seq, source_count)
    return [f"{index:06d}" for index in range(start, end)]


def _effective_mask_frame_index(handle: h5py.File, frame_name: str) -> int:
    frames = handle["frames"]
    if len(frames) == 0:
        raise ValueError("Effective-mask sidecar has no frames")
    first = frames[0]
    first_name = first.decode("utf-8") if isinstance(first, bytes) else str(first)
    if str(frame_name).isdigit() and first_name.isdigit():
        frame_index = int(frame_name) - int(first_name)
    else:
        stored = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in frames[:]]
        if str(frame_name) not in stored:
            raise ValueError(f"Effective-mask frame {frame_name!r} is absent from the sidecar")
        frame_index = stored.index(str(frame_name))
    if frame_index < 0 or frame_index >= len(frames):
        raise ValueError(f"Effective-mask frame {frame_name!r} is outside the sidecar timeline")
    stored_value = frames[frame_index]
    stored_name = stored_value.decode("utf-8") if isinstance(stored_value, bytes) else str(stored_value)
    if stored_name != str(frame_name):
        raise ValueError(f"Effective-mask frame mapping differs: requested {frame_name!r}, found {stored_name!r}")
    return frame_index


def _validate_effective_mask_sidecar(export_seq: str | Path, camera_id: int, sidecar: Path, handle: h5py.File) -> None:
    export_seq = Path(export_seq)
    expected_source_identities = raw_mask_identities(export_seq, camera_id)
    expected_pose_validity_identity = pose_validity_source_identity(export_seq)
    cache_key = (str(sidecar.resolve()), _file_identity(sidecar), export_seq.name, int(camera_id), _rgb_timeline_cache_identity(export_seq, camera_id), json.dumps(expected_source_identities, sort_keys=True, separators=(",", ":")), json.dumps(expected_pose_validity_identity, sort_keys=True, separators=(",", ":")))
    with _EFFECTIVE_MASK_VALIDATION_LOCK:
        if cache_key in _VALIDATED_EFFECTIVE_MASKS:
            _VALIDATED_EFFECTIVE_MASKS.move_to_end(cache_key)
            return
    expected_frames = frame_names(export_seq, camera_id)
    report = validate_effective_mask_handle(handle, expected_sequence=export_seq.name, expected_camera_id=camera_id, expected_frames=expected_frames, expected_source_identities=expected_source_identities)
    if report["metadata"].get("pose_validity_source_identity") != expected_pose_validity_identity:
        raise ValueError(f"Effective-mask pose-validity source identity mismatch: {report['metadata'].get('pose_validity_source_identity')} != {expected_pose_validity_identity}")
    with _EFFECTIVE_MASK_VALIDATION_LOCK:
        _VALIDATED_EFFECTIVE_MASKS[cache_key] = None
        _VALIDATED_EFFECTIVE_MASKS.move_to_end(cache_key)
        while len(_VALIDATED_EFFECTIVE_MASKS) > EFFECTIVE_MASK_VALIDATION_CACHE_MAX_ENTRIES:
            _VALIDATED_EFFECTIVE_MASKS.popitem(last=False)


def load_canonical_effective_mask_validity(export_seq: str | Path, effective_root: str | Path, camera_ids: Sequence[int], frames: Sequence[str]) -> dict[str, np.ndarray]:
    export_seq = Path(export_seq)
    frames = [str(frame) for frame in frames]
    camera_ids = [int(camera_id) for camera_id in camera_ids]
    if not frames or not camera_ids or len(camera_ids) != len(set(camera_ids)):
        raise ValueError("Canonical effective-mask validity requires nonempty frames and unique camera IDs")
    invalid_camera_ids = [camera_id for camera_id in camera_ids if camera_id < 0 or camera_id >= len(MHR_CAMERA_NAMES)]
    if invalid_camera_ids:
        raise ValueError(f"Canonical effective-mask validity has unsupported camera IDs: {invalid_camera_ids}")
    usable_by_camera = []
    object_nonempty_by_camera = []
    for camera_id in camera_ids:
        sidecar = Path(effective_root) / export_seq.name / f"{MHR_CAMERA_NAMES[camera_id]}.h5"
        if not sidecar.is_file():
            raise FileNotFoundError(f"Required effective-mask sidecar is missing for {export_seq.name}/{MHR_CAMERA_NAMES[camera_id]}: {sidecar}")
        with _borrow_h5(sidecar) as handle:
            _validate_effective_mask_sidecar(export_seq, camera_id, sidecar, handle)
            stored_frames = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in handle["frames"][()]]
            if stored_frames != frames:
                raise ValueError(f"Effective-mask and packed timelines differ for {export_seq.name}/{MHR_CAMERA_NAMES[camera_id]}")
            usability = effective_mask_frame_usability(handle)
            usable_by_camera.append(usability["frame_usable"])
            object_nonempty_by_camera.append(usability["object_nonempty"])
    frame_usable_by_camera = np.stack(usable_by_camera, axis=0).astype(bool, copy=False)
    return {"camera_ids": np.asarray(camera_ids, dtype=np.int16), "frame_usable_by_camera": frame_usable_by_camera, "frame_usable": np.all(frame_usable_by_camera, axis=0), "object_nonempty_by_camera": np.stack(object_nonempty_by_camera, axis=0).astype(bool, copy=False)}


def read_rgb(export_seq: str | Path, camera_id: int, frame_name: str) -> np.ndarray:
    from PIL import Image

    path = Path(export_seq) / "images" / MHR_CAMERA_NAMES[camera_id] / f"{frame_name}.png"
    if path.is_file():
        return np.asarray(Image.open(path).convert("RGB"))
    h5_path = _source_h5_path(export_seq, "images", camera_id)
    with _borrow_h5(h5_path) as handle:
        return read_rgb_h5_frame(handle, _frame_index(frame_name))


def read_raw_mask(export_seq: str | Path, kind: str, camera_id: int, frame_name: str) -> np.ndarray:
    from PIL import Image

    if kind not in {"human", "object"}:
        raise ValueError(f"Unsupported mask kind {kind}")
    dirname = "human_masks" if kind == "human" else "object_masks"
    path = Path(export_seq) / dirname / MHR_CAMERA_NAMES[camera_id] / f"{frame_name}.png"
    if path.is_file():
        return np.asarray(Image.open(path).convert("L")) > 127
    h5_path = _source_h5_path(export_seq, dirname, camera_id)
    with _borrow_h5(h5_path) as handle:
        return np.asarray(handle["frames"][_frame_index(frame_name)]) > 127


def read_mask_with_provenance(export_seq: str | Path, kind: str, camera_id: int, frame_name: str) -> EffectiveMask:
    raw = read_raw_mask(export_seq, kind, camera_id, frame_name)
    sidecar = _resolve_effective_mask_sidecar(export_seq, camera_id)
    if sidecar is None:
        return EffectiveMask(mask=raw, source=int(MASK_SOURCE_RAW))
    with _borrow_h5(sidecar) as handle:
        _validate_effective_mask_sidecar(export_seq, camera_id, sidecar, handle)
        result = read_effective_mask_from_handle(handle, kind, _effective_mask_frame_index(handle, frame_name), frame_name, raw)
    if kind == "object" and result.source == int(MASK_SOURCE_RAW_EMPTY_INVALID_POSE):
        raise UnusableObjectMaskError(f"Object mask is unusable because the raw mask is empty and ground-truth object pose is invalid: {Path(export_seq).name}/{MHR_CAMERA_NAMES[camera_id]}/{frame_name}")
    return result


def read_object_mask_with_provenance(export_seq: str | Path, camera_id: int, frame_name: str) -> EffectiveMask:
    return read_mask_with_provenance(export_seq, "object", camera_id, frame_name)


def read_object_mask(export_seq: str | Path, camera_id: int, frame_name: str) -> np.ndarray:
    return read_object_mask_with_provenance(export_seq, camera_id, frame_name).mask


def read_mask(export_seq: str | Path, kind: str, camera_id: int, frame_name: str) -> np.ndarray:
    return read_mask_with_provenance(export_seq, kind, camera_id, frame_name).mask


def read_frame_modalities(export_seq: str | Path, camera_name: str, frame_name: str, *, depth_root: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera_id = MHR_CAMERA_NAMES.index(str(camera_name))
    rgb = read_rgb(export_seq, camera_id, frame_name)
    depth = read_depth_m(export_seq, camera_id, frame_name, depth_root=depth_root)
    human = read_mask(export_seq, "human", camera_id, frame_name).astype(np.uint8) * 255
    object_mask = read_object_mask(export_seq, camera_id, frame_name).astype(np.uint8) * 255
    return rgb, depth, human, object_mask


def read_depth_m(
    export_seq: str | Path,
    camera_id: int,
    frame_name: str,
    *,
    depth_root: str | Path | None = None,
) -> np.ndarray:
    from PIL import Image

    export_seq = Path(export_seq)
    root = Path(depth_root) if depth_root is not None else export_seq / "depth"
    if root.is_file():
        return read_metric_depth(root, "aligned", MHR_CAMERA_NAMES[camera_id], frame_name)
    path = root / MHR_CAMERA_NAMES[camera_id] / f"{frame_name}.png"
    if path.is_file():
        depth = np.asarray(Image.open(path)).astype(np.float32)
    else:
        h5_path = _source_h5_path(export_seq, "depth", camera_id)
        with _borrow_h5(h5_path) as handle:
            depth = np.asarray(read_sensor_depth_h5_frame(handle, _frame_index(frame_name)), dtype=np.float32)
    if _is_raw_export_depth_root(export_seq, depth_root):
        return depthimage_png_to_depth_m(depth)
    return depth / 1000.0


def write_depth_png(depth_m: np.ndarray, path: str | Path) -> None:
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_mm = np.clip(depth_m * 1000.0, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(depth_mm).save(path)


def depth_to_points(depth_m: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    h, w = depth_m.shape[:2]
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    valid = depth_m > 0.001
    if mask is not None:
        valid &= mask.astype(bool)
    z = depth_m[valid]
    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    x = (xs[valid] - K[0, 2]) * z / K[0, 0]
    y = (ys[valid] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1).astype(np.float32)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points(points_cam: np.ndarray, K: np.ndarray, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape
    z = points_cam[:, 2]
    valid = z > 0.001
    uv = np.zeros((len(points_cam), 2), dtype=np.float32)
    uv[:, 0] = points_cam[:, 0] * K[0, 0] / np.maximum(z, 1e-8) + K[0, 2]
    uv[:, 1] = points_cam[:, 1] * K[1, 1] / np.maximum(z, 1e-8) + K[1, 2]
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, valid
