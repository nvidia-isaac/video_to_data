from __future__ import annotations

import errno
import io
import fcntl
import json
import os
import threading
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

import h5py
import numpy as np
from h5py import _objects as _h5py_objects


H5_CACHE_MAX_HANDLES = 32
DEPTH_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DEPTH_H5_FORMAT = "cari4d_mhr_metric_depth_png_v1"
DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE = "depth_alignment_input_identity_json"
SOURCE_LOCK_SUFFIX = ".storage-repack.lock"
_H5PY_FORK_STATE_ATTRIBUTE = "_cari4d_h5py_fork_state_v1"


def _h5py_fork_before() -> None:
    _h5py_objects.phil.acquire()


def _h5py_fork_parent() -> None:
    _h5py_objects.phil.release()


def _h5py_fork_child() -> None:
    state = getattr(_h5py_objects, _H5PY_FORK_STATE_ATTRIBUTE)
    if not state["native_phil_guard"]:
        _h5py_objects.phil.release()
    for reset in tuple(state["child_resets"].values()):
        reset()


def _h5py_fork_state() -> dict:
    state = getattr(_h5py_objects, _H5PY_FORK_STATE_ATTRIBUTE, None)
    if state is not None:
        return state
    native_phil_guard = hasattr(_h5py_objects, "_phil_before_fork") and hasattr(_h5py_objects, "_phil_after_fork")
    state = {"native_phil_guard": native_phil_guard, "child_resets": {}, "registration_count": 1}
    setattr(_h5py_objects, _H5PY_FORK_STATE_ATTRIBUTE, state)
    if native_phil_guard:
        os.register_at_fork(after_in_child=_h5py_fork_child)
    else:
        os.register_at_fork(before=_h5py_fork_before, after_in_parent=_h5py_fork_parent, after_in_child=_h5py_fork_child)
    return state


def _register_h5py_fork_child_reset(name: str, reset: Callable[[], None]) -> None:
    _h5py_fork_state()["child_resets"][str(name)] = reset


@dataclass
class _DepthCacheEntry:
    identity: tuple[int, int, int, int]
    handle: h5py.File
    owner_pid: int
    pins: int = 0


@dataclass(frozen=True)
class DepthFrameRecord:
    index: int
    raw_depth_m: np.ndarray
    aligned_depth_m: np.ndarray
    scale: float
    shift: float
    valid_count: int
    raw_payload: np.ndarray | None = None


@dataclass(frozen=True)
class _DepthWriteMetadata:
    index: int
    scale: float
    shift: float
    valid_count: int
    raw_shape: tuple[int, ...]
    aligned_shape: tuple[int, ...]


@dataclass(frozen=True)
class _DepthBatchSubmission:
    camera_name: str
    metadata: tuple[_DepthWriteMetadata, ...]
    payloads: tuple[tuple[np.ndarray, np.ndarray], ...] | None
    futures: tuple[Future[tuple[np.ndarray, np.ndarray]], ...] | None


_READ_HANDLES: OrderedDict[str, _DepthCacheEntry] = OrderedDict()
_FRAME_INDICES: dict[tuple[str, str], dict[str, int]] = {}
_CACHE_PID = os.getpid()
_CACHE_CONDITION = threading.Condition(threading.RLock())


def _after_fork_child() -> None:
    global _READ_HANDLES, _FRAME_INDICES, _CACHE_PID, _CACHE_CONDITION
    for entry in _READ_HANDLES.values():
        if entry.handle.id.valid:
            entry.handle.close()
    _READ_HANDLES = OrderedDict()
    _FRAME_INDICES = {}
    _CACHE_PID = os.getpid()
    _CACHE_CONDITION = threading.Condition(threading.RLock())


_register_h5py_fork_child_reset("mhr-depth-h5", _after_fork_child)


def _encode_depth_png_uint16(depth_mm: np.ndarray) -> np.ndarray:
    from PIL import Image

    value = np.asarray(depth_mm)
    if value.dtype != np.dtype("uint16") or value.ndim != 2:
        raise TypeError(f"Depth PNG input must be a two-dimensional uint16 array, got shape={value.shape}, dtype={value.dtype}")
    stream = io.BytesIO()
    Image.fromarray(value).save(stream, format="PNG", compress_level=9, optimize=True)
    return np.frombuffer(stream.getvalue(), dtype=np.uint8).copy()


def _encode_metric_depth(depth_m: np.ndarray) -> np.ndarray:
    depth_mm = np.clip(np.asarray(depth_m, dtype=np.float32) * 1000.0, 0.0, np.iinfo(np.uint16).max).astype(np.uint16)
    return _encode_depth_png_uint16(depth_mm)


def _encode_depth_record(record: DepthFrameRecord) -> tuple[np.ndarray, np.ndarray]:
    if record.raw_payload is None:
        raw_payload = _encode_metric_depth(record.raw_depth_m)
    else:
        raw_payload = np.asarray(record.raw_payload, dtype=np.uint8)
        raw_shape = _canonical_depth_shape(raw_payload)
        if raw_shape is None or raw_shape != tuple(np.asarray(record.raw_depth_m).shape):
            raise ValueError(f"Preserved raw depth payload is not canonical or differs from decoded shape: payload={raw_shape}, decoded={np.asarray(record.raw_depth_m).shape}")
    return raw_payload, _encode_metric_depth(record.aligned_depth_m)


def decode_depth_png_uint16(payload: np.ndarray) -> np.ndarray:
    from PIL import Image

    value = np.asarray(payload)
    if value.dtype != np.dtype("uint8") or value.ndim != 1:
        raise TypeError(f"Depth payload must be a one-dimensional uint8 array, got shape={value.shape}, dtype={value.dtype}")
    encoded = value.tobytes()
    if len(encoded) < 33 or encoded[:8] != DEPTH_PNG_SIGNATURE or int.from_bytes(encoded[8:12], "big") != 13 or encoded[12:16] != b"IHDR" or encoded[24] != 16 or encoded[25] != 0:
        raise ValueError("Generated depth payload must be a 16-bit grayscale PNG")
    with Image.open(io.BytesIO(encoded)) as image:
        if image.format != "PNG":
            raise ValueError(f"Generated depth payload must be a 16-bit grayscale PNG, got {image.format}")
        image.load()
        decoded = np.asarray(image)
    if decoded.ndim != 2 or not np.issubdtype(decoded.dtype, np.integer):
        raise TypeError(f"Generated depth PNG must decode to a two-dimensional uint16-compatible array, got shape={decoded.shape}, dtype={decoded.dtype}")
    if decoded.size and (int(decoded.min()) < 0 or int(decoded.max()) > np.iinfo(np.uint16).max):
        raise ValueError(f"Generated depth PNG values are outside uint16 range: min={int(decoded.min())}, max={int(decoded.max())}")
    return decoded.astype(np.uint16, copy=False)


def _decode_metric_depth(payload: np.ndarray) -> np.ndarray:
    return decode_depth_png_uint16(payload).astype(np.float32) / 1000.0


def _canonical_depth_shape(payload: np.ndarray) -> tuple[int, int] | None:
    try:
        decoded = decode_depth_png_uint16(payload)
    except (OSError, TypeError, ValueError):
        return None
    if not np.array_equal(np.asarray(payload), _encode_depth_png_uint16(decoded)):
        return None
    return int(decoded.shape[0]), int(decoded.shape[1])


def _decode_names(values: np.ndarray) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _decode_json_attribute(value: object, name: str) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    decoded = json.loads(str(value))
    if not isinstance(decoded, dict):
        raise TypeError(f"{name} must decode to a JSON object")
    return decoded


def _file_identity(path: str | Path) -> tuple[int, int, int, int]:
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def depth_h5_file_identity(path: str | Path) -> dict[str, int]:
    device, inode, size, mtime_ns = _file_identity(path)
    return {"device": device, "inode": inode, "size": size, "mtime_ns": mtime_ns}


def _source_lock_path(path: str | Path) -> Path:
    source_path = Path(path).resolve()
    return source_path.with_name(f".{source_path.name}{SOURCE_LOCK_SUFFIX}")


def _open_shared_source_lock(path: str | Path):
    source_path = Path(path).resolve()
    lock_path = _source_lock_path(source_path)
    try:
        descriptor = os.open(lock_path, os.O_RDONLY | os.O_CLOEXEC)
    except FileNotFoundError:
        try:
            created_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o666)
        except FileExistsError:
            pass
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EPERM, errno.EROFS}:
                raise
            descriptor = os.open(source_path, os.O_RDONLY | os.O_CLOEXEC)
            return os.fdopen(descriptor, "rb")
        else:
            os.close(created_descriptor)
        descriptor = os.open(lock_path, os.O_RDONLY | os.O_CLOEXEC)
    return os.fdopen(descriptor, "rb")


@contextmanager
def _shared_source_lock(path: str | Path) -> Iterator[None]:
    with _open_shared_source_lock(path) as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _clear_frame_indices(path_key: str) -> None:
    for cache_key in [cache_key for cache_key in _FRAME_INDICES if cache_key[0] == path_key]:
        del _FRAME_INDICES[cache_key]


def _open_stable_read_handle(path_key: str) -> tuple[tuple[int, int, int, int], h5py.File]:
    for _ in range(3):
        before = _file_identity(path_key)
        handle = h5py.File(path_key, "r")
        after = _file_identity(path_key)
        if before == after:
            return after, handle
        handle.close()
    raise RuntimeError(f"HDF5 path changed repeatedly while opening: {path_key}")


def _close_cache_entry(path_key: str, entry: _DepthCacheEntry) -> None:
    if entry.handle.id.valid:
        entry.handle.close()
    _clear_frame_indices(path_key)


def _reset_cache_pid_locked() -> None:
    global _CACHE_PID
    pid = os.getpid()
    if _CACHE_PID == pid:
        return
    for path_key, entry in list(_READ_HANDLES.items()):
        _close_cache_entry(path_key, entry)
    _READ_HANDLES.clear()
    _FRAME_INDICES.clear()
    _CACHE_PID = pid


def _evict_unpinned_locked() -> bool:
    for path_key, entry in list(_READ_HANDLES.items()):
        if entry.pins == 0:
            del _READ_HANDLES[path_key]
            _close_cache_entry(path_key, entry)
            return True
    return False


@contextmanager
def _borrow_cached_read_handle(path: str | Path) -> Iterator[h5py.File]:
    path_key = str(Path(path).resolve())
    entry = None
    with _CACHE_CONDITION:
        while entry is None:
            _reset_cache_pid_locked()
            identity = _file_identity(path_key)
            cached = _READ_HANDLES.get(path_key)
            if cached is not None and cached.identity == identity and cached.handle.id.valid:
                cached.pins += 1
                _READ_HANDLES.move_to_end(path_key)
                entry = cached
                break
            if cached is not None:
                if cached.pins:
                    _CACHE_CONDITION.wait()
                    continue
                del _READ_HANDLES[path_key]
                _close_cache_entry(path_key, cached)
            retry = False
            while len(_READ_HANDLES) >= H5_CACHE_MAX_HANDLES:
                if _evict_unpinned_locked():
                    continue
                _CACHE_CONDITION.wait()
                retry = True
                break
            if retry:
                continue
            identity, handle = _open_stable_read_handle(path_key)
            entry = _DepthCacheEntry(identity=identity, handle=handle, owner_pid=os.getpid(), pins=1)
            _READ_HANDLES[path_key] = entry
    try:
        yield entry.handle
    finally:
        with _CACHE_CONDITION:
            entry.pins -= 1
            _CACHE_CONDITION.notify_all()


@contextmanager
def _borrow_read_handle(path: str | Path) -> Iterator[h5py.File]:
    with _shared_source_lock(path):
        with _borrow_cached_read_handle(path) as handle:
            yield handle


def clear_depth_h5_cache() -> None:
    with _CACHE_CONDITION:
        _reset_cache_pid_locked()
        while any(entry.pins for entry in _READ_HANDLES.values()):
            _CACHE_CONDITION.wait()
        for path_key, entry in list(_READ_HANDLES.items()):
            _close_cache_entry(path_key, entry)
        _READ_HANDLES.clear()
        _FRAME_INDICES.clear()
        _CACHE_CONDITION.notify_all()


def depth_frame_names(path: str | Path, camera_name: str) -> list[str]:
    with _borrow_read_handle(path) as handle:
        return _decode_names(handle[f"frame_names/{camera_name}"][:])


def _frame_indices_for_handle(path_key: str, camera_name: str, handle: h5py.File) -> dict[str, int]:
    cache_key = (path_key, str(camera_name))
    with _CACHE_CONDITION:
        indices = _FRAME_INDICES.get(cache_key)
    if indices is not None:
        return indices
    names = _decode_names(handle[f"frame_names/{camera_name}"][:])
    loaded = {name: index for index, name in enumerate(names)}
    if len(loaded) != len(names):
        raise ValueError(f"Depth H5 frame names contain duplicates for {path_key} camera {camera_name}")
    with _CACHE_CONDITION:
        return _FRAME_INDICES.setdefault(cache_key, loaded)


def _frame_index(path: str | Path, camera_name: str, frame_name: str) -> int:
    path_key = str(Path(path).resolve())
    with _borrow_read_handle(path_key) as handle:
        indices = _frame_indices_for_handle(path_key, camera_name, handle)
        try:
            return indices[str(frame_name)]
        except KeyError as exc:
            raise KeyError(f"Frame {frame_name} is absent from {path} camera {camera_name}") from exc


def read_metric_depth(path: str | Path, kind: str, camera_name: str, frame_name: str) -> np.ndarray:
    if kind not in {"raw", "aligned"}:
        raise ValueError(f"Unsupported depth kind {kind}")
    path_key = str(Path(path).resolve())
    with _borrow_read_handle(path_key) as handle:
        indices = _frame_indices_for_handle(path_key, camera_name, handle)
        try:
            index = indices[str(frame_name)]
        except KeyError as exc:
            raise KeyError(f"Frame {frame_name} is absent from {path} camera {camera_name}") from exc
        return _decode_metric_depth(handle[f"{kind}/{camera_name}"][index])


def _validated_depth_record_shape(raw_payload: np.ndarray, aligned_payload: np.ndarray) -> tuple[int, int] | None:
    raw_shape = _canonical_depth_shape(raw_payload)
    aligned_shape = _canonical_depth_shape(aligned_payload)
    return raw_shape if raw_shape is not None and raw_shape == aligned_shape else None


def validate_depth_h5(path: str | Path, *, expected_cameras: Sequence[str] | None = None, expected_alignment_method: str | None = None, expected_alignment_input_identity: Mapping[str, object] | None = None, use_lock: bool = True, validation_workers: int = 1, validation_batch_size: int = 32, validate_payloads: bool = True) -> dict[str, object]:
    if validation_workers <= 0 or validation_batch_size <= 0:
        raise ValueError(f"validation_workers and validation_batch_size must be positive, got {validation_workers}, {validation_batch_size}")
    path = Path(path).resolve()
    lock_context = _shared_source_lock(path) if use_lock else nullcontext()
    executor = ThreadPoolExecutor(max_workers=validation_workers) if validate_payloads and validation_workers > 1 else None
    with lock_context:
        try:
            with h5py.File(path, "r") as handle:
                file_format = handle.attrs.get("format")
                if isinstance(file_format, (bytes, np.bytes_)):
                    file_format = bytes(file_format).decode("utf-8")
                if file_format != DEPTH_H5_FORMAT:
                    raise ValueError(f"Depth H5 format must be {DEPTH_H5_FORMAT!r}, got {file_format!r}")
                if not bool(handle.attrs.get("complete", False)):
                    raise ValueError(f"Depth H5 is incomplete: {path}")
                alignment_method = handle.attrs.get("depth_alignment_method")
                if isinstance(alignment_method, (bytes, np.bytes_)):
                    alignment_method = bytes(alignment_method).decode("utf-8")
                if expected_alignment_method is not None and alignment_method != expected_alignment_method:
                    raise ValueError(f"Depth H5 alignment method differs: expected={expected_alignment_method!r}, actual={alignment_method!r}")
                alignment_input_identity = _decode_json_attribute(handle.attrs.get(DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE), DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE)
                if expected_alignment_input_identity is not None and alignment_input_identity != dict(expected_alignment_input_identity):
                    raise ValueError("Depth H5 alignment input identity differs from the current depth and effective-mask inputs")
                for group_name in ("frame_names", "raw", "aligned", "alignment"):
                    if group_name not in handle or not isinstance(handle[group_name], h5py.Group):
                        raise TypeError(f"Depth H5 node /{group_name} must be a group")
                cameras = sorted(handle["frame_names"])
                if expected_cameras is not None and cameras != sorted(str(camera) for camera in expected_cameras):
                    raise ValueError(f"Depth H5 cameras differ: expected={sorted(expected_cameras)}, actual={cameras}")
                if set(handle["raw"]) != set(cameras) or set(handle["aligned"]) != set(cameras) or set(handle["alignment"]) != set(cameras):
                    raise ValueError("Depth H5 camera groups do not match")
                frame_counts = {}
                frame_shapes = {}
                for camera_name in cameras:
                    names_dataset = handle[f"frame_names/{camera_name}"]
                    if not isinstance(names_dataset, h5py.Dataset) or names_dataset.ndim != 1 or h5py.check_string_dtype(names_dataset.dtype) is None:
                        raise TypeError(f"Depth H5 frame_names/{camera_name} schema must be one-dimensional strings")
                    names = _decode_names(names_dataset[:])
                    if len(set(names)) != len(names):
                        raise ValueError(f"Depth H5 frame names contain duplicates for {camera_name}")
                    frame_counts[camera_name] = len(names)
                    expected_shape = (len(names),)
                    for kind in ("raw", "aligned"):
                        dataset = handle[f"{kind}/{camera_name}"]
                        if not isinstance(dataset, h5py.Dataset) or dataset.shape != expected_shape or dataset.ndim != 1 or h5py.check_dtype(vlen=dataset.dtype) != np.dtype("uint8"):
                            raise TypeError(f"Depth H5 {kind}/{camera_name} schema must be one-dimensional variable-length uint8 with shape {expected_shape}")
                    alignment = handle[f"alignment/{camera_name}"]
                    if not isinstance(alignment, h5py.Group):
                        raise TypeError(f"Depth H5 alignment/{camera_name} must be a group")
                    for key, dtype in (("scale", "float32"), ("shift", "float32"), ("valid_count", "int32")):
                        if key not in alignment or not isinstance(alignment[key], h5py.Dataset) or alignment[key].shape != expected_shape or alignment[key].dtype != np.dtype(dtype):
                            raise TypeError(f"Depth H5 alignment/{camera_name}/{key} schema must have shape {expected_shape} and dtype {dtype}")
                    scales = np.asarray(alignment["scale"][:], dtype=np.float32)
                    shifts = np.asarray(alignment["shift"][:], dtype=np.float32)
                    valid_counts = np.asarray(alignment["valid_count"][:], dtype=np.int32)
                    valid_metadata = np.isfinite(scales) & np.isfinite(shifts) & (valid_counts >= 0)
                    if not np.all(valid_metadata):
                        raise ValueError(f"Depth H5 has invalid alignment metadata at {camera_name}[{int(np.flatnonzero(~valid_metadata)[0])}]")
                    if not validate_payloads:
                        frame_shapes[camera_name] = None
                        continue
                    shapes = []
                    for start in range(0, len(names), validation_batch_size):
                        stop = min(start + validation_batch_size, len(names))
                        raw_payloads = handle[f"raw/{camera_name}"][start:stop]
                        aligned_payloads = handle[f"aligned/{camera_name}"][start:stop]
                        inputs = list(zip(raw_payloads, aligned_payloads))
                        batch_shapes = list(executor.map(lambda values: _validated_depth_record_shape(*values), inputs)) if executor is not None else [_validated_depth_record_shape(*values) for values in inputs]
                        invalid = next((index for index, shape in enumerate(batch_shapes) if shape is None), None)
                        if invalid is not None:
                            raise ValueError(f"Depth H5 has an invalid record at {camera_name}[{start + invalid}]")
                        shapes.extend(batch_shapes)
                    if shapes and len(set(shapes)) != 1:
                        raise ValueError(f"Depth H5 frame shapes are inconsistent for {camera_name}: {shapes}")
                    frame_shapes[camera_name] = shapes[0] if shapes else None
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
    return {"path": str(path), "format": DEPTH_H5_FORMAT, "alignment_method": alignment_method, "alignment_input_identity": alignment_input_identity, "cameras": cameras, "frame_counts": frame_counts, "frame_shapes": frame_shapes, "validation_mode": "exhaustive" if validate_payloads else "structural"}


def validate_depth_h5_structure(path: str | Path, *, expected_cameras: Sequence[str] | None = None, expected_alignment_method: str | None = None, expected_alignment_input_identity: Mapping[str, object] | None = None, use_lock: bool = True, validation_workers: int = 1, validation_batch_size: int = 32) -> dict[str, object]:
    return validate_depth_h5(path, expected_cameras=expected_cameras, expected_alignment_method=expected_alignment_method, expected_alignment_input_identity=expected_alignment_input_identity, use_lock=use_lock, validation_workers=validation_workers, validation_batch_size=validation_batch_size, validate_payloads=False)


def merge_depth_h5_shards(output_path: str | Path, shard_paths: Sequence[str | Path], *, expected_cameras: Sequence[str] | None = None, expected_alignment_method: str | None = None, expected_alignment_input_identity: Mapping[str, object] | None = None, redo: bool = False, remove_shards: bool = False, validation_workers: int = 32, validation_batch_size: int = 32, validation_mode: str = "exhaustive") -> dict[str, object]:
    output_path = Path(output_path).resolve()
    shards = [Path(path).resolve() for path in shard_paths]
    if not shards:
        raise ValueError("At least one depth H5 shard is required")
    if len(set(shards)) != len(shards):
        raise ValueError("Depth H5 shard paths must be unique")
    if validation_mode not in {"exhaustive", "structural"}:
        raise ValueError(f"Depth H5 validation mode must be exhaustive or structural, got {validation_mode!r}")
    validator = validate_depth_h5 if validation_mode == "exhaustive" else validate_depth_h5_structure
    if output_path.exists() and not redo:
        return validator(output_path, expected_cameras=expected_cameras, expected_alignment_method=expected_alignment_method, expected_alignment_input_identity=expected_alignment_input_identity)
    if validation_workers <= 0 or validation_batch_size <= 0:
        raise ValueError(f"validation_workers and validation_batch_size must be positive, got {validation_workers}, {validation_batch_size}")
    shard_workers = min(len(shards), validation_workers)
    workers_per_shard = max(1, validation_workers // shard_workers)
    with ThreadPoolExecutor(max_workers=shard_workers) as executor:
        reports = list(executor.map(lambda path: validator(path, expected_alignment_method=expected_alignment_method, validation_workers=workers_per_shard, validation_batch_size=validation_batch_size), shards))
    cameras = [camera for report in reports for camera in report["cameras"]]
    if len(cameras) != len(set(cameras)):
        raise ValueError(f"Depth H5 shards contain duplicate cameras: {cameras}")
    if expected_cameras is not None and sorted(cameras) != sorted(str(camera) for camera in expected_cameras):
        raise ValueError(f"Depth H5 shard cameras differ: expected={sorted(expected_cameras)}, actual={sorted(cameras)}")
    methods = {report["alignment_method"] for report in reports}
    if len(methods) != 1:
        raise ValueError(f"Depth H5 shards use different alignment methods: {methods}")
    identities = [report["alignment_input_identity"] for report in reports]
    alignment_input_identity = None
    if any(identity is not None for identity in identities):
        if any(identity is None for identity in identities):
            raise ValueError("Depth H5 shards mix missing and populated alignment input identities")
        common_identities = [{key: value for key, value in identity.items() if key != "cameras"} for identity in identities]
        if any(value != common_identities[0] for value in common_identities[1:]):
            raise ValueError(f"Depth H5 shard alignment input identities disagree outside camera-specific inputs: {common_identities}")
        camera_identities = {}
        for identity in identities:
            for camera_name, camera_identity in dict(identity.get("cameras", {})).items():
                if camera_name in camera_identities:
                    raise ValueError(f"Depth H5 shards contain duplicate alignment input identity for {camera_name}")
                camera_identities[camera_name] = camera_identity
        alignment_input_identity = {**common_identities[0], "cameras": camera_identities}
        if expected_alignment_input_identity is not None and alignment_input_identity != dict(expected_alignment_input_identity):
            raise ValueError("Merged depth H5 alignment input identity differs from the expected identity")
    elif expected_alignment_input_identity is not None:
        raise ValueError("Depth H5 shards are missing the expected alignment input identity")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.merge-{os.getpid()}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        with h5py.File(temporary_path, "w") as target:
            target.attrs["format"] = DEPTH_H5_FORMAT
            target.attrs["complete"] = False
            if next(iter(methods)) is not None:
                target.attrs["depth_alignment_method"] = next(iter(methods))
            if alignment_input_identity is not None:
                target.attrs[DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE] = json.dumps(alignment_input_identity, sort_keys=True, separators=(",", ":"))
            for group_name in ("frame_names", "raw", "aligned", "alignment"):
                target.create_group(group_name)
            for shard in shards:
                with _shared_source_lock(shard):
                    with h5py.File(shard, "r") as source:
                        for camera_name in source["frame_names"]:
                            for group_name in ("frame_names", "raw", "aligned", "alignment"):
                                source.copy(source[f"{group_name}/{camera_name}"], target[group_name], name=camera_name)
            target.attrs.modify("complete", True)
            target.flush()
        report = validator(temporary_path, expected_cameras=expected_cameras, expected_alignment_method=expected_alignment_method, expected_alignment_input_identity=expected_alignment_input_identity, use_lock=False, validation_workers=validation_workers, validation_batch_size=validation_batch_size)
        temporary_identity = _file_identity(temporary_path)
        os.replace(temporary_path, output_path)
        if _file_identity(output_path) != temporary_identity:
            raise RuntimeError(f"Atomic depth H5 publish changed file identity: {output_path}")
        report = {**report, "path": str(output_path)}
        if remove_shards:
            for shard in shards:
                shard.unlink()
                _source_lock_path(shard).unlink(missing_ok=True)
        return report
    finally:
        temporary_path.unlink(missing_ok=True)


def merge_depth_h5_replacement_shards(output_path: str | Path, shard_paths: Sequence[str | Path], *, expected_cameras: Sequence[str], expected_alignment_method: str, expected_alignment_input_identity: Mapping[str, object], effective_mask_identity_equivalences: Mapping[str, Mapping[str, Mapping[str, str]]] | None = None, remove_shards: bool = False, validation_workers: int = 16, validation_batch_size: int = 32) -> dict[str, object]:
    output_path = Path(output_path).resolve()
    shards = [Path(path).resolve() for path in shard_paths]
    expected_cameras = sorted(str(camera) for camera in expected_cameras)
    expected_alignment_input_identity = dict(expected_alignment_input_identity)
    effective_mask_identity_equivalences = dict(effective_mask_identity_equivalences or {})
    if not output_path.is_file() or not shards:
        raise FileNotFoundError(f"Partial depth replacement requires an existing output and at least one shard: output={output_path.is_file()} shards={len(shards)}")
    if len(set(shards)) != len(shards) or output_path in shards:
        raise ValueError("Partial depth replacement shard paths must be unique and differ from the output")
    if validation_workers <= 0 or validation_batch_size <= 0:
        raise ValueError(f"validation_workers and validation_batch_size must be positive, got {validation_workers}, {validation_batch_size}")
    source_report = validate_depth_h5_structure(output_path, expected_cameras=expected_cameras, expected_alignment_method=expected_alignment_method)
    shard_reports = [validate_depth_h5_structure(path, expected_alignment_method=expected_alignment_method) for path in shards]
    replacement_cameras = [camera for report in shard_reports for camera in report["cameras"]]
    if not replacement_cameras or len(replacement_cameras) != len(set(replacement_cameras)):
        raise ValueError(f"Partial depth replacement shards contain duplicate or no cameras: {replacement_cameras}")
    if not set(replacement_cameras).issubset(expected_cameras):
        raise ValueError(f"Partial depth replacement cameras are outside the expected set: replacements={replacement_cameras}, expected={expected_cameras}")
    source_identity = source_report["alignment_input_identity"]
    if source_identity is None or any(report["alignment_input_identity"] is None for report in shard_reports):
        raise ValueError("Partial depth replacement requires source and shard alignment input identities")
    expected_common_identity = {key: value for key, value in expected_alignment_input_identity.items() if key != "cameras"}
    source_common_identity = {key: value for key, value in source_identity.items() if key != "cameras"}
    if source_common_identity != expected_common_identity:
        raise ValueError(f"Partial depth replacement source identity differs outside camera-specific inputs: stored={source_common_identity}, expected={expected_common_identity}")
    expected_identity_cameras = dict(expected_alignment_input_identity.get("cameras", {}))
    if sorted(expected_identity_cameras) != expected_cameras:
        raise ValueError("Partial depth replacement expected identity cameras differ from the expected camera set")
    camera_sources: dict[str, Path] = {camera: output_path for camera in expected_cameras}
    for shard, report in zip(shards, shard_reports):
        identity = report["alignment_input_identity"]
        shard_common_identity = {key: value for key, value in identity.items() if key != "cameras"}
        if shard_common_identity != expected_common_identity:
            raise ValueError(f"Partial depth replacement shard identity differs outside camera-specific inputs for {shard}: stored={shard_common_identity}, expected={expected_common_identity}")
        for camera in report["cameras"]:
            if identity["cameras"][camera] != expected_identity_cameras[camera]:
                raise ValueError(f"Partial depth replacement shard camera identity differs for {camera}")
            camera_sources[camera] = shard
    source_identity_cameras = dict(source_identity.get("cameras", {}))
    for camera in set(expected_cameras) - set(replacement_cameras):
        source_camera, expected_camera = source_identity_cameras.get(camera), expected_identity_cameras[camera]
        if not isinstance(source_camera, dict):
            raise ValueError(f"Partial depth replacement source identity is missing {camera}")
        normalized = json.loads(json.dumps(source_camera))
        for kind in ("human_mask", "object_mask"):
            source_effective = normalized.get(kind, {}).get("effective")
            expected_effective = expected_camera.get(kind, {}).get("effective")
            if not isinstance(source_effective, dict) or not isinstance(expected_effective, dict):
                raise ValueError(f"Unchanged depth camera {camera} has an invalid effective {kind} identity")
            source_sha256, expected_sha256 = source_effective.get("logical_sha256"), expected_effective.get("logical_sha256")
            if source_sha256 != expected_sha256:
                equivalence = effective_mask_identity_equivalences.get(camera, {}).get(kind)
                expected_equivalence = {"source_logical_sha256": source_sha256, "target_logical_sha256": expected_sha256}
                if not isinstance(equivalence, Mapping) or dict(equivalence) != expected_equivalence:
                    raise ValueError(f"Unchanged depth camera {camera} has a different effective {kind} identity without an exact numerical-equivalence certificate")
            normalized[kind]["effective"] = expected_effective
        if normalized != expected_camera:
            raise ValueError(f"Unchanged depth camera {camera} input identity changed outside effective-mask metadata")
    temporary_path = output_path.with_name(f".{output_path.name}.partial-merge-{os.getpid()}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        with h5py.File(temporary_path, "w") as target:
            target.attrs["format"] = DEPTH_H5_FORMAT
            target.attrs["complete"] = False
            target.attrs["depth_alignment_method"] = expected_alignment_method
            target.attrs[DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE] = json.dumps(expected_alignment_input_identity, sort_keys=True, separators=(",", ":"))
            for group_name in ("frame_names", "raw", "aligned", "alignment"):
                target.create_group(group_name)
            for camera in expected_cameras:
                source_path = camera_sources[camera]
                with _shared_source_lock(source_path):
                    with h5py.File(source_path, "r") as source:
                        for group_name in ("frame_names", "raw", "aligned", "alignment"):
                            source.copy(source[f"{group_name}/{camera}"], target[group_name], name=camera)
            target.attrs.modify("complete", True)
            target.flush()
        report = validate_depth_h5_structure(temporary_path, expected_cameras=expected_cameras, expected_alignment_method=expected_alignment_method, expected_alignment_input_identity=expected_alignment_input_identity, use_lock=False, validation_workers=validation_workers, validation_batch_size=validation_batch_size)
        temporary_identity = _file_identity(temporary_path)
        clear_depth_h5_cache()
        os.replace(temporary_path, output_path)
        if _file_identity(output_path) != temporary_identity:
            raise RuntimeError(f"Atomic partial depth publish changed file identity: {output_path}")
        if remove_shards:
            for shard in shards:
                shard.unlink()
                _source_lock_path(shard).unlink(missing_ok=True)
        return {**report, "path": str(output_path), "replaced_cameras": sorted(replacement_cameras)}
    finally:
        temporary_path.unlink(missing_ok=True)


class MHRDepthH5Writer:
    def __init__(self, path: str | Path, frame_names_by_camera: Mapping[str, Sequence[str]], *, redo: bool = False, alignment_method: str | None = None, alignment_input_identity: Mapping[str, object] | None = None, encoding_workers: int = 1, max_pending_batches: int = 2):
        self.path = Path(path)
        self.alignment_method = None if alignment_method is None else str(alignment_method)
        self.alignment_input_identity = None if alignment_input_identity is None else dict(alignment_input_identity)
        if self.alignment_method == "":
            raise ValueError("Depth alignment method must be a nonempty string")
        self.encoding_workers = int(encoding_workers)
        self.max_pending_batches = int(max_pending_batches)
        if self.encoding_workers <= 0 or self.max_pending_batches <= 0:
            raise ValueError("encoding_workers and max_pending_batches must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = None
        self._lock_handle = None
        self._encoding_executor: ThreadPoolExecutor | None = None
        self._pending_batches: deque[_DepthBatchSubmission] = deque()
        self._mutation_prepared = False
        self._validated_shapes: dict[tuple[str, int], tuple[int, int]] = {}
        lock_path = _source_lock_path(self.path)
        self._lock_handle = lock_path.open("a+")
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            clear_depth_h5_cache()
            if redo and self.path.exists():
                self.path.unlink()
            self.handle = h5py.File(self.path, "a")
            self._initialize(frame_names_by_camera)
        except BaseException:
            self.close()
            raise

    def _prepare_mutation(self) -> None:
        if self._mutation_prepared:
            return
        self.handle.attrs["complete"] = False
        self.handle.flush()
        self._mutation_prepared = True

    def _require_group(self, path: str) -> h5py.Group:
        if path in self.handle:
            group = self.handle[path]
            if not isinstance(group, h5py.Group):
                raise TypeError(f"Depth H5 node /{path} must be a group")
            return group
        self._prepare_mutation()
        return self.handle.create_group(path)

    def _initialize(self, frame_names_by_camera: Mapping[str, Sequence[str]]) -> None:
        if "complete" not in self.handle.attrs:
            self._prepare_mutation()
        if "format" not in self.handle.attrs:
            self._prepare_mutation()
            self.handle.attrs["format"] = DEPTH_H5_FORMAT
        else:
            existing_format = self.handle.attrs["format"]
            if isinstance(existing_format, (bytes, np.bytes_)):
                existing_format = bytes(existing_format).decode("utf-8")
            if existing_format != DEPTH_H5_FORMAT:
                raise ValueError(f"Depth H5 format must be {DEPTH_H5_FORMAT!r}, got {existing_format!r}")
        existing_alignment_method = self.handle.attrs.get("depth_alignment_method")
        if isinstance(existing_alignment_method, (bytes, np.bytes_)):
            existing_alignment_method = bytes(existing_alignment_method).decode("utf-8")
        if existing_alignment_method is not None and self.alignment_method is not None and existing_alignment_method != self.alignment_method:
            raise ValueError(f"Depth H5 alignment method is {existing_alignment_method!r}, not {self.alignment_method!r}; use redo to rebuild the file")
        existing_alignment_input_identity = _decode_json_attribute(self.handle.attrs.get(DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE), DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE)
        if existing_alignment_input_identity is not None and self.alignment_input_identity is not None and existing_alignment_input_identity != self.alignment_input_identity:
            raise ValueError("Depth H5 alignment input identity differs; use redo to rebuild the file")
        payload_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        string_dtype = h5py.string_dtype(encoding="utf-8")
        for camera_name, names_seq in frame_names_by_camera.items():
            names = [str(name) for name in names_seq]
            frame_group = self._require_group("frame_names")
            if camera_name in frame_group:
                existing = _decode_names(frame_group[camera_name][:])
                if existing != names:
                    raise ValueError(f"Depth H5 frame names changed for {camera_name}")
            else:
                self._prepare_mutation()
                frame_group.create_dataset(camera_name, data=np.asarray(names, dtype=object), dtype=string_dtype)
            for kind in ("raw", "aligned"):
                group = self._require_group(kind)
                if camera_name not in group:
                    self._prepare_mutation()
                    group.create_dataset(camera_name, shape=(len(names),), dtype=payload_dtype)
                elif group[camera_name].shape != (len(names),):
                    raise ValueError(f"Depth H5 {kind}/{camera_name} shape changed")
            alignment = self._require_group(f"alignment/{camera_name}")
            for key, dtype, fillvalue in (("scale", "float32", np.nan), ("shift", "float32", np.nan), ("valid_count", "int32", -1)):
                if key not in alignment:
                    self._prepare_mutation()
                    alignment.create_dataset(key, shape=(len(names),), dtype=dtype, fillvalue=fillvalue)
                elif alignment[key].shape != (len(names),) or alignment[key].dtype != np.dtype(dtype):
                    raise ValueError(f"Depth H5 alignment/{camera_name}/{key} schema changed")
        self._validate_schema()
        if self.alignment_method is not None and existing_alignment_method is None:
            populated = any(np.any(self.handle[f"alignment/{camera_name}/valid_count"][:] >= 0) for camera_name in self.handle["frame_names"])
            if populated:
                raise ValueError("Populated depth H5 is missing depth_alignment_method; use redo to rebuild the file")
            self._prepare_mutation()
            self.handle.attrs["depth_alignment_method"] = self.alignment_method
            self.handle.flush()
        if self.alignment_input_identity is not None and existing_alignment_input_identity is None:
            populated = any(np.any(self.handle[f"alignment/{camera_name}/valid_count"][:] >= 0) for camera_name in self.handle["frame_names"])
            if populated:
                raise ValueError("Populated depth H5 is missing depth alignment input identity; use the metadata migration or realignment path")
            self._prepare_mutation()
            self.handle.attrs[DEPTH_ALIGNMENT_INPUT_IDENTITY_ATTRIBUTE] = json.dumps(self.alignment_input_identity, sort_keys=True, separators=(",", ":"))
            self.handle.flush()

    def _validate_schema(self) -> None:
        for group_name in ("frame_names", "raw", "aligned", "alignment"):
            if group_name not in self.handle or not isinstance(self.handle[group_name], h5py.Group):
                raise TypeError(f"Depth H5 node /{group_name} must be a group")
        cameras = set(self.handle["frame_names"])
        for group_name in ("raw", "aligned", "alignment"):
            if set(self.handle[group_name]) != cameras:
                raise ValueError(f"Depth H5 /{group_name} cameras do not match /frame_names")
        for camera_name in cameras:
            names_dataset = self.handle[f"frame_names/{camera_name}"]
            if not isinstance(names_dataset, h5py.Dataset) or names_dataset.ndim != 1 or h5py.check_string_dtype(names_dataset.dtype) is None:
                raise TypeError(f"Depth H5 frame_names/{camera_name} schema must be one-dimensional strings")
            names = _decode_names(names_dataset[:])
            if len(set(names)) != len(names):
                raise ValueError(f"Depth H5 frame_names/{camera_name} contains duplicates")
            expected_shape = (len(names),)
            for kind in ("raw", "aligned"):
                dataset = self.handle[f"{kind}/{camera_name}"]
                if not isinstance(dataset, h5py.Dataset) or dataset.shape != expected_shape or dataset.ndim != 1 or h5py.check_dtype(vlen=dataset.dtype) != np.dtype("uint8"):
                    raise TypeError(f"Depth H5 {kind}/{camera_name} schema must be one-dimensional variable-length uint8 with shape {expected_shape}")
            alignment = self.handle[f"alignment/{camera_name}"]
            if not isinstance(alignment, h5py.Group):
                raise TypeError(f"Depth H5 alignment/{camera_name} must be a group")
            for key, dtype in (("scale", "float32"), ("shift", "float32"), ("valid_count", "int32")):
                if key not in alignment or not isinstance(alignment[key], h5py.Dataset) or alignment[key].shape != expected_shape or alignment[key].dtype != np.dtype(dtype):
                    raise TypeError(f"Depth H5 alignment/{camera_name}/{key} schema must have shape {expected_shape} and dtype {dtype}")

    def __enter__(self) -> "MHRDepthH5Writer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            if exc_type is None:
                self.flush_submitted()
        finally:
            self.close()

    def _valid_frame_shape(self, camera_name: str, index: int) -> tuple[int, int] | None:
        cache_key = (str(camera_name), int(index))
        if cache_key in self._validated_shapes:
            return self._validated_shapes[cache_key]
        scale = float(self.handle[f"alignment/{camera_name}/scale"][index])
        shift = float(self.handle[f"alignment/{camera_name}/shift"][index])
        valid_count = int(self.handle[f"alignment/{camera_name}/valid_count"][index])
        raw_payload = self.handle[f"raw/{camera_name}"][index]
        aligned_payload = self.handle[f"aligned/{camera_name}"][index]
        raw_shape = _canonical_depth_shape(raw_payload)
        aligned_shape = _canonical_depth_shape(aligned_payload)
        if raw_shape is None or raw_shape != aligned_shape or not np.isfinite(scale) or not np.isfinite(shift) or valid_count < 0:
            return None
        self._validated_shapes[cache_key] = raw_shape
        return raw_shape

    def has_frame(self, camera_name: str, index: int) -> bool:
        return self._valid_frame_shape(camera_name, index) is not None

    def write_frame(self, camera_name: str, index: int, raw_depth_m: np.ndarray, aligned_depth_m: np.ndarray, *, scale: float, shift: float, valid_count: int) -> None:
        self.write_frames(camera_name, [DepthFrameRecord(index, raw_depth_m, aligned_depth_m, scale, shift, valid_count)])

    def _validate_records(self, camera_name: str, records: Sequence[DepthFrameRecord]) -> tuple[list[DepthFrameRecord], tuple[_DepthWriteMetadata, ...]]:
        records = list(records)
        if not records:
            return records, ()
        indices = [int(record.index) for record in records]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Depth frame batch contains duplicate indices: {indices}")
        frame_count = len(self.handle[f"frame_names/{camera_name}"])
        if any(index < 0 or index >= frame_count for index in indices):
            raise IndexError(f"Depth frame batch indices must be inside [0, {frame_count}): {indices}")
        metadata = tuple(_DepthWriteMetadata(int(record.index), float(record.scale), float(record.shift), int(record.valid_count), tuple(np.asarray(record.raw_depth_m).shape), tuple(np.asarray(record.aligned_depth_m).shape)) for record in records)
        return records, metadata

    def _executor(self, encoding_workers: int | None) -> ThreadPoolExecutor | None:
        workers = self.encoding_workers if encoding_workers is None else int(encoding_workers)
        if workers <= 0:
            raise ValueError(f"encoding_workers must be positive, got {workers}")
        if workers == 1:
            return None
        if self._encoding_executor is None:
            self.encoding_workers = workers
            self._encoding_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mhr-depth-png")
        elif workers != self.encoding_workers:
            raise ValueError(f"A depth writer must use one persistent encoding worker count, got {workers} after {self.encoding_workers}")
        return self._encoding_executor

    def _submit_batch(self, camera_name: str, records: Sequence[DepthFrameRecord], encoding_workers: int | None) -> _DepthBatchSubmission | None:
        records, metadata = self._validate_records(camera_name, records)
        if not records:
            return None
        executor = self._executor(encoding_workers)
        if executor is None:
            return _DepthBatchSubmission(str(camera_name), metadata, tuple(_encode_depth_record(record) for record in records), None)
        return _DepthBatchSubmission(str(camera_name), metadata, None, tuple(executor.submit(_encode_depth_record, record) for record in records))

    @staticmethod
    def _resolve_payloads(submission: _DepthBatchSubmission) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        if submission.payloads is not None:
            return submission.payloads
        if submission.futures is None:
            raise RuntimeError("Depth batch submission contains neither payloads nor futures")
        return tuple(future.result() for future in submission.futures)

    def _commit_submission(self, submission: _DepthBatchSubmission) -> None:
        payloads = self._resolve_payloads(submission)
        camera_name = submission.camera_name
        indices = [metadata.index for metadata in submission.metadata]
        self._prepare_mutation()
        empty_payload = np.empty((0,), dtype=np.uint8)
        for index in indices:
            self._validated_shapes.pop((camera_name, index), None)
            self.handle[f"alignment/{camera_name}/valid_count"][index] = np.int32(-1)
            self.handle[f"alignment/{camera_name}/scale"][index] = np.float32(np.nan)
            self.handle[f"alignment/{camera_name}/shift"][index] = np.float32(np.nan)
            self.handle[f"raw/{camera_name}"][index] = empty_payload
            self.handle[f"aligned/{camera_name}"][index] = empty_payload
        self.handle.flush()
        for metadata, (raw_payload, aligned_payload) in zip(submission.metadata, payloads):
            index = metadata.index
            self.handle[f"raw/{camera_name}"][index] = raw_payload
            self.handle[f"aligned/{camera_name}"][index] = aligned_payload
            self.handle[f"alignment/{camera_name}/scale"][index] = np.float32(metadata.scale)
            self.handle[f"alignment/{camera_name}/shift"][index] = np.float32(metadata.shift)
            self.handle[f"alignment/{camera_name}/valid_count"][index] = np.int32(metadata.valid_count)
            if len(metadata.raw_shape) == 2 and metadata.raw_shape == metadata.aligned_shape and np.isfinite(metadata.scale) and np.isfinite(metadata.shift) and metadata.valid_count >= 0:
                self._validated_shapes[(camera_name, index)] = metadata.raw_shape
        self.handle.flush()

    def submit_frames(self, camera_name: str, records: Sequence[DepthFrameRecord], *, encoding_workers: int | None = None) -> None:
        while len(self._pending_batches) >= self.max_pending_batches:
            self._commit_submission(self._pending_batches.popleft())
        submission = self._submit_batch(camera_name, records, encoding_workers)
        if submission is not None:
            self._pending_batches.append(submission)

    def flush_submitted(self) -> None:
        while self._pending_batches:
            self._commit_submission(self._pending_batches.popleft())

    def write_frames(self, camera_name: str, records: Sequence[DepthFrameRecord], *, encoding_workers: int | None = None) -> None:
        self.flush_submitted()
        submission = self._submit_batch(camera_name, records, encoding_workers)
        if submission is not None:
            self._commit_submission(submission)

    def mark_complete(self) -> None:
        self.flush_submitted()
        for camera_name in self.handle["frame_names"]:
            shapes = [self._valid_frame_shape(camera_name, index) for index in range(len(self.handle[f"frame_names/{camera_name}"]))]
            missing = [index for index, shape in enumerate(shapes) if shape is None]
            if missing:
                self._prepare_mutation()
                raise ValueError(f"Cannot mark depth H5 complete; {camera_name} has {len(missing)} missing or invalid records")
            if shapes and len(set(shapes)) != 1:
                self._prepare_mutation()
                raise ValueError(f"Cannot mark depth H5 complete; {camera_name} frame shapes are not internally consistent: {shapes}")
        if not bool(self.handle.attrs.get("complete", False)):
            self.handle.attrs["complete"] = True
            self.handle.flush()
        self._mutation_prepared = False

    def close(self) -> None:
        self._pending_batches.clear()
        if self._encoding_executor is not None:
            self._encoding_executor.shutdown(wait=True, cancel_futures=True)
            self._encoding_executor = None
        if self.handle:
            self.handle.flush()
            self.handle.close()
            self.handle = None
        if self._lock_handle is not None:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
            self._lock_handle.close()
            self._lock_handle = None
