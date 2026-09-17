from __future__ import annotations

import fcntl
import hashlib
import os
import threading
from copy import deepcopy
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


RGB = "rgb"
SENSOR_DEPTH = "sensor-depth"
RGB_FFV1_SCHEMA = "cari4d.rgb_ffv1_sidecar.v1"
SENSOR_DEPTH_FFV1_SCHEMA = "cari4d.sensor_depth_ffv1_sidecar.v1"
SCHEMA_TO_KIND = {RGB_FFV1_SCHEMA: RGB, SENSOR_DEPTH_FFV1_SCHEMA: SENSOR_DEPTH}
SOURCE_LOCK_SUFFIX = ".storage-repack.lock"
FFV1_CACHE_MAX_DECODERS = 16
FFV1_CACHE_MAX_METADATA = 128
HASH_BLOCK_BYTES = 64 * 1024 * 1024


def _text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: str | Path) -> tuple[int, int, int, int]:
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _source_lock_path(path: str | Path) -> Path:
    source = Path(path).resolve()
    return source.with_name(f".{source.name}{SOURCE_LOCK_SUFFIX}")


@contextmanager
def _shared_source_lock(path: str | Path) -> Iterator[None]:
    lock_path = _source_lock_path(path)
    lock_path.touch(exist_ok=True)
    with lock_path.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ffv1_sidecar_kind(path: str | Path | h5py.Group) -> str | None:
    if isinstance(path, h5py.Group):
        schema = _text(path.attrs.get("schema", ""))
        return SCHEMA_TO_KIND.get(schema)
    with h5py.File(path, "r") as handle:
        return ffv1_sidecar_kind(handle)


def _required_attributes(kind: str) -> dict[str, Any]:
    if kind == RGB:
        return {"schema": RGB_FFV1_SCHEMA, "encoding": "ffv1", "container": "matroska", "frame_rate": 30, "channels": 3, "source_dtype": "uint8", "encoded_pixel_format": "bgr0", "ffv1_level": 3, "ffv1_coder": 1, "ffv1_context": 1, "ffv1_slicecrc": 1, "gop": 1}
    if kind == SENSOR_DEPTH:
        return {"schema": SENSOR_DEPTH_FFV1_SCHEMA, "encoding": "ffv1", "container": "matroska", "frame_rate": 30, "source_dtype": "uint16", "encoded_pixel_format": "gray16le", "ffv1_level": 3, "ffv1_coder": 1, "ffv1_context": 1, "ffv1_slicecrc": 1, "gop": 32}
    raise ValueError(f"Unsupported FFV1 sidecar kind: {kind!r}")


def _attribute_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        return _text(actual) == expected
    if isinstance(expected, bool):
        return isinstance(actual, (bool, np.bool_)) and bool(actual) == expected
    return isinstance(actual, (int, np.integer)) and not isinstance(actual, (bool, np.bool_)) and int(actual) == expected


@dataclass
class _MetadataEntry:
    metadata_identity: tuple[int, int, int, int]
    sidecar_identity: tuple[int, int, int, int]
    metadata: dict[str, Any]
    owner_pid: int


_METADATA: OrderedDict[str, _MetadataEntry] = OrderedDict()
_METADATA_CACHE_PID = os.getpid()
_METADATA_CACHE_LOCK = threading.RLock()


def _reset_metadata_pid() -> None:
    global _METADATA_CACHE_PID
    if _METADATA_CACHE_PID != os.getpid():
        _METADATA.clear()
        _METADATA_CACHE_PID = os.getpid()


def _cached_metadata(metadata_path: Path, expected_kind: str | None) -> dict[str, Any] | None:
    path_key = str(metadata_path)
    with _METADATA_CACHE_LOCK:
        _reset_metadata_pid()
        entry = _METADATA.get(path_key)
        if entry is None:
            return None
        try:
            unchanged = entry.metadata_identity == _file_identity(metadata_path) and entry.sidecar_identity == _file_identity(entry.metadata["sidecar_path"])
        except FileNotFoundError:
            unchanged = False
        if not unchanged:
            del _METADATA[path_key]
            return None
        if expected_kind is not None and entry.metadata["kind"] != expected_kind:
            raise ValueError(f"FFV1 metadata kind differs: expected={expected_kind}, actual={entry.metadata['kind']}")
        _METADATA.move_to_end(path_key)
        return deepcopy(entry.metadata)


def _cache_metadata(metadata_path: Path, metadata: dict[str, Any]) -> None:
    path_key = str(metadata_path)
    entry = _MetadataEntry(_file_identity(metadata_path), _file_identity(metadata["sidecar_path"]), deepcopy(metadata), os.getpid())
    with _METADATA_CACHE_LOCK:
        _reset_metadata_pid()
        _METADATA[path_key] = entry
        _METADATA.move_to_end(path_key)
        while len(_METADATA) > FFV1_CACHE_MAX_METADATA:
            _METADATA.popitem(last=False)


def validate_ffv1_metadata(path: str | Path, expected_kind: str | None = None, verify_digest: bool = False, use_lock: bool = True) -> dict[str, Any]:
    metadata_path = Path(path).resolve()
    lock_context = _shared_source_lock(metadata_path) if use_lock else nullcontext()
    with lock_context:
        if not verify_digest:
            cached = _cached_metadata(metadata_path, expected_kind)
            if cached is not None:
                return cached
        with h5py.File(metadata_path, "r") as handle:
            kind = ffv1_sidecar_kind(handle)
            if kind is None:
                raise ValueError(f"H5 file is not canonical FFV1 metadata: {metadata_path}")
            if expected_kind is not None and kind != expected_kind:
                raise ValueError(f"FFV1 metadata kind differs: expected={expected_kind}, actual={kind}")
            for key, expected in _required_attributes(kind).items():
                if key not in handle.attrs or not _attribute_equal(handle.attrs[key], expected):
                    raise ValueError(f"FFV1 metadata attribute {key!r} is invalid: expected={expected!r}, actual={handle.attrs.get(key)!r}")
            if "complete" not in handle.attrs or not _attribute_equal(handle.attrs["complete"], True):
                raise ValueError(f"FFV1 metadata is incomplete: {metadata_path}")
            frame_count = int(handle.attrs.get("frame_count", -1))
            height = int(handle.attrs.get("height", -1))
            width = int(handle.attrs.get("width", -1))
            if frame_count <= 0 or height <= 0 or width <= 0:
                raise ValueError(f"FFV1 metadata dimensions are invalid: frames={frame_count}, height={height}, width={width}")
            if "frame_pts" not in handle or "keyframe_indices" not in handle:
                raise KeyError(f"FFV1 metadata lacks frame_pts or keyframe_indices: {metadata_path}")
            frame_pts = np.asarray(handle["frame_pts"][:], dtype=np.int64)
            keyframe_indices = np.asarray(handle["keyframe_indices"][:], dtype=np.int64)
            if frame_pts.shape != (frame_count,) or np.any(np.diff(frame_pts) <= 0):
                raise ValueError(f"FFV1 frame_pts must be strictly increasing with shape {(frame_count,)}, got {frame_pts.shape}")
            if keyframe_indices.ndim != 1 or len(keyframe_indices) == 0 or int(keyframe_indices[0]) != 0 or np.any(np.diff(keyframe_indices) <= 0) or int(keyframe_indices[-1]) >= frame_count:
                raise ValueError(f"FFV1 keyframe indices are invalid: {keyframe_indices.tolist()}")
            gop = int(handle.attrs["gop"])
            if np.any(np.diff(keyframe_indices) > gop) or frame_count - int(keyframe_indices[-1]) > gop:
                raise ValueError(f"FFV1 keyframe interval exceeds GOP {gop}: {keyframe_indices.tolist()}")
            if kind == RGB and (len(keyframe_indices) != frame_count or np.any(keyframe_indices != np.arange(frame_count))):
                raise ValueError("RGB FFV1 GOP-1 metadata must mark every frame as a keyframe")
            sidecar_basename = _text(handle.attrs.get("sidecar_basename", ""))
            if not sidecar_basename or Path(sidecar_basename).name != sidecar_basename:
                raise ValueError(f"FFV1 sidecar basename is invalid: {sidecar_basename!r}")
            sidecar_path = metadata_path.parent / sidecar_basename
            if not sidecar_path.is_file():
                raise FileNotFoundError(sidecar_path)
            sidecar_sha256 = _text(handle.attrs.get("sidecar_sha256", ""))
            if not _is_sha256(sidecar_sha256):
                raise ValueError(f"FFV1 sidecar SHA-256 is invalid: {sidecar_sha256!r}")
            logical_stem = _text(handle.attrs.get("logical_stem", metadata_path.stem))
            if not logical_stem or Path(logical_stem).name != logical_stem:
                raise ValueError(f"FFV1 logical stem is invalid: {logical_stem!r}")
            if sidecar_basename != f"{logical_stem}.ffv1.{sidecar_sha256}.mkv":
                raise ValueError(f"FFV1 sidecar basename does not bind the metadata stem and SHA-256: {sidecar_basename}")
            sidecar_bytes = int(handle.attrs.get("sidecar_bytes", -1))
            if sidecar_bytes <= 0 or sidecar_path.stat().st_size != sidecar_bytes:
                raise ValueError(f"FFV1 sidecar byte count differs: metadata={sidecar_bytes}, actual={sidecar_path.stat().st_size}")
            if verify_digest and file_sha256(sidecar_path) != sidecar_sha256:
                raise ValueError(f"FFV1 sidecar SHA-256 differs: {sidecar_path}")
            source_h5_bytes = int(handle.attrs.get("source_h5_bytes", -1))
            if source_h5_bytes <= 0:
                raise ValueError(f"FFV1 source H5 byte count is invalid: {source_h5_bytes}")
            result = {"kind": kind, "schema": _text(handle.attrs["schema"]), "metadata_path": str(metadata_path), "sidecar_path": str(sidecar_path), "sidecar_basename": sidecar_basename, "sidecar_sha256": sidecar_sha256, "sidecar_bytes": sidecar_bytes, "source_h5_bytes": source_h5_bytes, "frame_count": frame_count, "height": height, "width": width, "gop": gop, "encoded_pixel_format": _text(handle.attrs["encoded_pixel_format"]), "source_layout": _text(handle.attrs.get("source_layout", "")), "exact_to_original_dense": bool(handle.attrs.get("exact_to_original_dense", kind == SENSOR_DEPTH)), "frame_pts": frame_pts.tolist(), "keyframe_indices": keyframe_indices.tolist()}
            if "pre_jpeg_original_logical_sha256" in handle.attrs:
                result["pre_jpeg_original_logical_sha256"] = _text(handle.attrs["pre_jpeg_original_logical_sha256"])
            if "jpeg_published_logical_sha256" in handle.attrs:
                result["jpeg_published_logical_sha256"] = _text(handle.attrs["jpeg_published_logical_sha256"])
            if "source_logical_sha256" in handle.attrs:
                result["source_logical_sha256"] = _text(handle.attrs["source_logical_sha256"])
            _cache_metadata(metadata_path, result)
            return deepcopy(result)


@dataclass
class _DecoderEntry:
    metadata_identity: tuple[int, int, int, int]
    sidecar_identity: tuple[int, int, int, int]
    container: Any
    stream: Any
    iterator: Any
    last_index: int
    owner_pid: int
    lock: threading.RLock


_DECODERS: OrderedDict[str, _DecoderEntry] = OrderedDict()
_DECODER_CACHE_PID = os.getpid()
_DECODER_CACHE_LOCK = threading.RLock()


def _close_decoder(entry: _DecoderEntry) -> None:
    entry.container.close()


def clear_ffv1_cache() -> None:
    global _DECODER_CACHE_PID, _METADATA_CACHE_PID
    with _DECODER_CACHE_LOCK:
        for entry in _DECODERS.values():
            _close_decoder(entry)
        _DECODERS.clear()
        _DECODER_CACHE_PID = os.getpid()
    with _METADATA_CACHE_LOCK:
        _METADATA.clear()
        _METADATA_CACHE_PID = os.getpid()


def _reset_decoder_pid() -> None:
    global _DECODER_CACHE_PID
    if _DECODER_CACHE_PID != os.getpid():
        clear_ffv1_cache()


os.register_at_fork(after_in_child=clear_ffv1_cache)


def _decoder_entry(metadata: dict[str, Any]) -> _DecoderEntry:
    import av

    metadata_path = metadata["metadata_path"]
    sidecar_path = metadata["sidecar_path"]
    metadata_identity = _file_identity(metadata_path)
    sidecar_identity = _file_identity(sidecar_path)
    with _DECODER_CACHE_LOCK:
        _reset_decoder_pid()
        entry = _DECODERS.get(metadata_path)
        if entry is not None and entry.metadata_identity == metadata_identity and entry.sidecar_identity == sidecar_identity:
            _DECODERS.move_to_end(metadata_path)
            return entry
        if entry is not None:
            _close_decoder(entry)
            del _DECODERS[metadata_path]
        while len(_DECODERS) >= FFV1_CACHE_MAX_DECODERS:
            _, evicted = _DECODERS.popitem(last=False)
            _close_decoder(evicted)
        container = av.open(sidecar_path, mode="r")
        if len(container.streams.video) != 1:
            container.close()
            raise ValueError(f"FFV1 sidecar must contain one video stream: {sidecar_path}")
        stream = container.streams.video[0]
        entry = _DecoderEntry(metadata_identity, sidecar_identity, container, stream, None, -1, os.getpid(), threading.RLock())
        _DECODERS[metadata_path] = entry
        return entry


def _decoded_array(frame: Any, metadata: dict[str, Any]) -> np.ndarray:
    if metadata["kind"] == RGB:
        value = np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8)
        expected = (metadata["height"], metadata["width"], 3)
    else:
        value = np.asarray(frame.to_ndarray(format="gray16le"), dtype=np.uint16)
        expected = (metadata["height"], metadata["width"])
    if value.shape != expected:
        raise ValueError(f"FFV1 decoded frame shape differs: expected={expected}, actual={value.shape}")
    return value


def read_ffv1_frame(path: str | Path, index: int) -> np.ndarray:
    metadata = validate_ffv1_metadata(path, verify_digest=False)
    if not isinstance(index, (int, np.integer)) or not 0 <= int(index) < metadata["frame_count"]:
        raise IndexError(f"FFV1 frame index {index!r} is outside [0, {metadata['frame_count']})")
    index = int(index)
    entry = _decoder_entry(metadata)
    with entry.lock:
        if entry.iterator is not None and index == entry.last_index + 1:
            frame = next(entry.iterator)
            if int(frame.pts) != int(metadata["frame_pts"][index]):
                raise ValueError(f"Sequential FFV1 presentation timestamp differs at frame {index}: expected={metadata['frame_pts'][index]}, actual={frame.pts}")
            entry.last_index = index
            return _decoded_array(frame, metadata).copy()
        target_pts = int(metadata["frame_pts"][index])
        entry.container.seek(target_pts, stream=entry.stream, any_frame=False, backward=True)
        entry.iterator = iter(entry.container.decode(entry.stream))
        for frame in entry.iterator:
            frame_pts = int(frame.pts)
            if frame_pts < target_pts:
                continue
            if frame_pts != target_pts:
                raise ValueError(f"FFV1 seek skipped target frame {index}: expected PTS={target_pts}, actual={frame_pts}")
            entry.last_index = index
            return _decoded_array(frame, metadata).copy()
        raise EOFError(f"FFV1 seek reached end of stream before frame {index}: {metadata['sidecar_path']}")


def validate_ffv1_payload(path: str | Path, verify_digest: bool = True) -> dict[str, Any]:
    import av

    metadata = validate_ffv1_metadata(path, verify_digest=verify_digest)
    digest = hashlib.sha256()
    decoded_pts = []
    decoded_keyframes = []
    with av.open(metadata["sidecar_path"], mode="r") as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"FFV1 sidecar must contain one video stream: {metadata['sidecar_path']}")
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index >= metadata["frame_count"]:
                raise ValueError(f"FFV1 sidecar contains more than {metadata['frame_count']} frames: {metadata['sidecar_path']}")
            value = _decoded_array(frame, metadata)
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(str(value.shape).encode("utf-8"))
            digest.update(np.ascontiguousarray(value).tobytes())
            if frame.pts is None:
                raise ValueError(f"FFV1 frame {index} lacks a presentation timestamp")
            decoded_pts.append(int(frame.pts))
            if bool(frame.key_frame):
                decoded_keyframes.append(index)
    if len(decoded_pts) != metadata["frame_count"]:
        raise ValueError(f"FFV1 decoded frame count differs: expected={metadata['frame_count']}, actual={len(decoded_pts)}")
    if decoded_pts != metadata["frame_pts"]:
        raise ValueError(f"FFV1 decoded presentation timestamps differ: {metadata['sidecar_path']}")
    if decoded_keyframes != metadata["keyframe_indices"]:
        raise ValueError(f"FFV1 decoded keyframes differ: expected={metadata['keyframe_indices']}, actual={decoded_keyframes}")
    logical_sha256 = digest.hexdigest()
    if logical_sha256 != metadata.get("source_logical_sha256"):
        raise ValueError(f"FFV1 decoded logical SHA-256 differs: expected={metadata.get('source_logical_sha256')}, actual={logical_sha256}")
    return {**metadata, "decoded_logical_sha256": logical_sha256, "decoded_frame_count": len(decoded_pts)}
