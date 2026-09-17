from __future__ import annotations

import io
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


EFFECTIVE_MASK_SCHEMA = "cari4d.mhr_effective_masks.v1"
EFFECTIVE_MASK_REVISION = "pose-valid-raw-nonempty-else-joint-gt-visible-render-v2"
EFFECTIVE_MASK_WINDOW_VALIDITY_REVISION = "all-sampled-cameras-no-invalid-pose-empty-or-both-empty-v1"
MASK_KINDS = ("human", "object")
MASK_SOURCE_RAW = np.uint8(0)
MASK_SOURCE_GT_RENDER = np.uint8(1)
MASK_SOURCE_GT_RENDER_EMPTY = np.uint8(2)
MASK_SOURCE_RAW_EMPTY_INVALID_POSE = np.uint8(3)
MASK_SOURCE_NAMES = {
    int(MASK_SOURCE_RAW): "raw",
    int(MASK_SOURCE_GT_RENDER): "gt_visible_render",
    int(MASK_SOURCE_GT_RENDER_EMPTY): "gt_visible_render_empty",
    int(MASK_SOURCE_RAW_EMPTY_INVALID_POSE): "raw_empty_invalid_pose",
}


@dataclass(frozen=True)
class EffectiveMask:
    mask: np.ndarray
    source: int

    @property
    def source_name(self) -> str:
        return MASK_SOURCE_NAMES[int(self.source)]


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def path_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def effective_mask_kind_identity(path: str | Path, kind: str) -> dict[str, Any]:
    if kind not in MASK_KINDS:
        raise ValueError(f"Unsupported mask kind {kind!r}")
    path = Path(path)
    with h5py.File(path, "r") as handle:
        schema = _decode_text(handle.attrs.get("schema", ""))
        revision = _decode_text(handle.attrs.get("revision", ""))
        complete = bool(handle.attrs.get("complete", False))
        logical_sha256 = _decode_text(handle.attrs.get(f"{kind}_logical_sha256", ""))
    if schema != EFFECTIVE_MASK_SCHEMA or revision != EFFECTIVE_MASK_REVISION or not complete or len(logical_sha256) != 64:
        raise ValueError(f"Invalid effective-mask identity at {path}: schema={schema!r} revision={revision!r} complete={complete} {kind}_sha256={logical_sha256!r}")
    return {"path": str(path.resolve()), "schema": schema, "revision": revision, "logical_sha256": logical_sha256}


def encode_binary_mask(mask: np.ndarray) -> np.ndarray:
    from PIL import Image

    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError(f"Binary mask must have shape [H,W], got {value.shape}")
    if value.dtype != np.dtype("bool"):
        if not np.isin(value, (0, 1)).all():
            raise ValueError("Binary mask contains values outside 0/1")
        value = value.astype(bool)
    stream = io.BytesIO()
    Image.fromarray(value.astype(np.uint8) * 255, mode="L").save(stream, format="PNG", compress_level=1)
    return np.frombuffer(stream.getvalue(), dtype=np.uint8).copy()


def decode_binary_mask(payload: np.ndarray) -> np.ndarray:
    from PIL import Image

    value = np.asarray(payload)
    if value.dtype != np.dtype("uint8") or value.ndim != 1:
        raise TypeError(f"Mask payload must be one-dimensional uint8, got shape={value.shape}, dtype={value.dtype}")
    with Image.open(io.BytesIO(value.tobytes())) as image:
        if image.format != "PNG":
            raise ValueError(f"Effective mask payload must be PNG, got {image.format}")
        decoded = np.asarray(image.convert("L"))
    if decoded.ndim != 2 or not np.isin(decoded, (0, 255)).all():
        raise ValueError(f"Effective mask PNG must decode to a binary [H,W] image, got {decoded.shape}")
    return decoded > 127


def validate_effective_mask_handle(handle: h5py.File, *, expected_sequence: str | None = None, expected_camera_id: int | None = None, expected_frames: Sequence[str] | None = None, expected_source_identities: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    schema = _decode_text(handle.attrs.get("schema", ""))
    revision = _decode_text(handle.attrs.get("revision", ""))
    if schema != EFFECTIVE_MASK_SCHEMA or revision != EFFECTIVE_MASK_REVISION:
        raise ValueError(f"Unsupported effective-mask schema/revision: {schema!r}/{revision!r}")
    if not bool(handle.attrs.get("complete", False)):
        raise ValueError("Effective-mask H5 is not marked complete")
    sequence = _decode_text(handle.attrs.get("sequence", ""))
    camera_id = int(handle.attrs.get("camera_id", -1))
    frames = [_decode_text(value) for value in handle["frames"][()]]
    if expected_sequence is not None and sequence != str(expected_sequence):
        raise ValueError(f"Effective-mask sequence mismatch: {sequence!r} != {expected_sequence!r}")
    if expected_camera_id is not None and camera_id != int(expected_camera_id):
        raise ValueError(f"Effective-mask camera mismatch: {camera_id} != {expected_camera_id}")
    if expected_frames is not None and frames != [str(frame) for frame in expected_frames]:
        raise ValueError("Effective-mask frame timeline differs from the source RGB timeline")
    metadata = json.loads(_decode_text(handle["metadata_json"][()]))
    if expected_source_identities is not None:
        actual = metadata.get("raw_mask_identities")
        if actual != dict(expected_source_identities):
            raise ValueError(f"Effective-mask raw source identity mismatch: {actual} != {dict(expected_source_identities)}")
    frame_count = len(frames)
    for kind in MASK_KINDS:
        provenance = np.asarray(handle[f"{kind}/provenance"][()], dtype=np.uint8)
        slots = np.asarray(handle[f"{kind}/replacement_slot"][()], dtype=np.int32)
        payloads = handle[f"{kind}/replacement_png"]
        if provenance.shape != (frame_count,) or slots.shape != (frame_count,):
            raise ValueError(f"Effective-mask {kind} provenance/slot shape differs from frame count {frame_count}")
        if not np.isin(provenance, tuple(MASK_SOURCE_NAMES)).all():
            raise ValueError(f"Effective-mask {kind} provenance contains an unsupported value")
        replacement = np.isin(provenance, (MASK_SOURCE_GT_RENDER, MASK_SOURCE_GT_RENDER_EMPTY))
        if np.any(slots[~replacement] != -1) or np.any(slots[replacement] < 0) or np.any(slots[replacement] >= len(payloads)):
            raise ValueError(f"Effective-mask {kind} replacement slots are inconsistent with provenance")
        if len(np.unique(slots[replacement])) != int(replacement.sum()) or set(slots[replacement].tolist()) != set(range(len(payloads))):
            raise ValueError(f"Effective-mask {kind} replacement slots are not a dense one-to-one mapping")
        if len(_decode_text(handle.attrs.get(f"{kind}_logical_sha256", ""))) != 64:
            raise ValueError(f"Effective-mask {kind} logical SHA-256 is missing")
    return {"sequence": sequence, "camera_id": camera_id, "frames": frames, "metadata": metadata}


def effective_mask_frame_usability(handle: h5py.File) -> dict[str, np.ndarray]:
    report = validate_effective_mask_handle(handle)
    human_source = np.asarray(handle["human/provenance"][()], dtype=np.uint8)
    object_source = np.asarray(handle["object/provenance"][()], dtype=np.uint8)
    human_usable = human_source != MASK_SOURCE_RAW_EMPTY_INVALID_POSE
    object_usable = object_source != MASK_SOURCE_RAW_EMPTY_INVALID_POSE
    human_empty = np.isin(human_source, (MASK_SOURCE_GT_RENDER_EMPTY, MASK_SOURCE_RAW_EMPTY_INVALID_POSE))
    object_empty = np.isin(object_source, (MASK_SOURCE_GT_RENDER_EMPTY, MASK_SOURCE_RAW_EMPTY_INVALID_POSE))
    both_empty = human_empty & object_empty
    frame_usable = human_usable & object_usable & ~both_empty
    if any(value.shape != (len(report["frames"]),) for value in (human_usable, object_usable, both_empty, frame_usable)):
        raise ValueError("Effective-mask frame usability arrays differ from the sidecar timeline")
    return {"human_usable": human_usable, "object_usable": object_usable, "human_empty": human_empty, "object_empty": object_empty, "object_nonempty": ~object_empty, "both_empty": both_empty, "frame_usable": frame_usable}


def read_effective_mask_from_handle(handle: h5py.File, kind: str, frame_index: int, frame_name: str, raw_mask: np.ndarray) -> EffectiveMask:
    if kind not in MASK_KINDS:
        raise ValueError(f"Unsupported mask kind {kind!r}")
    frame_index = int(frame_index)
    raw = np.asarray(raw_mask, dtype=bool)
    stored_frame = _decode_text(handle["frames"][frame_index])
    if stored_frame != str(frame_name):
        raise ValueError(f"Effective-mask frame mismatch at index {frame_index}: {stored_frame!r} != {frame_name!r}")
    source = int(handle[f"{kind}/provenance"][frame_index])
    slot = int(handle[f"{kind}/replacement_slot"][frame_index])
    if source == int(MASK_SOURCE_RAW):
        if slot != -1:
            raise ValueError(f"Raw {kind} mask has unexpected replacement slot {slot}")
        if not np.any(raw):
            raise ValueError(f"Effective-mask sidecar marks an empty raw {kind} mask as raw at frame {frame_name}")
        return EffectiveMask(mask=raw, source=source)
    if source == int(MASK_SOURCE_RAW_EMPTY_INVALID_POSE):
        if slot != -1 or np.any(raw):
            raise ValueError(f"Invalid retained raw-empty {kind} mask at frame {frame_name}: slot={slot} nonempty={bool(np.any(raw))}")
        return EffectiveMask(mask=raw, source=source)
    if np.any(raw):
        raise ValueError(f"Effective-mask sidecar attempts to replace a nonempty raw {kind} mask at frame {frame_name}")
    if source not in {int(MASK_SOURCE_GT_RENDER), int(MASK_SOURCE_GT_RENDER_EMPTY)} or slot < 0:
        raise ValueError(f"Invalid effective-mask provenance/slot for {kind} frame {frame_name}: {source}/{slot}")
    replacement = decode_binary_mask(np.asarray(handle[f"{kind}/replacement_png"][slot], dtype=np.uint8))
    if replacement.shape != raw.shape:
        raise ValueError(f"Effective {kind} mask shape differs from raw source at frame {frame_name}: {replacement.shape} != {raw.shape}")
    if source == int(MASK_SOURCE_GT_RENDER) and not np.any(replacement):
        raise ValueError(f"Effective {kind} mask is empty but provenance says gt_visible_render at frame {frame_name}")
    if source == int(MASK_SOURCE_GT_RENDER_EMPTY) and np.any(replacement):
        raise ValueError(f"Effective {kind} mask is nonempty but provenance says gt_visible_render_empty at frame {frame_name}")
    return EffectiveMask(mask=replacement, source=source)


def write_encoded_effective_masks(path: str | Path, *, sequence: str, camera_id: int, camera_name: str, frames: Sequence[str], encoded: Mapping[str, Mapping[str, Any]], raw_mask_identities: Mapping[str, Mapping[str, Any]], metadata: Mapping[str, Any] | None = None, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Effective-mask H5 already exists: {path}")
    frames = [str(frame) for frame in frames]
    frame_count = len(frames)
    if frame_count == 0:
        raise ValueError("Effective-mask H5 requires at least one frame")
    normalized = {}
    for kind in MASK_KINDS:
        provenance = np.asarray(encoded[kind]["provenance"], dtype=np.uint8)
        slots = np.asarray(encoded[kind]["replacement_slot"], dtype=np.int32)
        payloads = [np.asarray(payload, dtype=np.uint8) for payload in encoded[kind]["replacement_png"]]
        if provenance.shape != (frame_count,) or slots.shape != (frame_count,):
            raise ValueError(f"Encoded {kind} provenance/slot must have shape {(frame_count,)}")
        if not np.isin(provenance, tuple(MASK_SOURCE_NAMES)).all():
            raise ValueError(f"Encoded {kind} provenance contains an unsupported value")
        replacement = np.isin(provenance, (MASK_SOURCE_GT_RENDER, MASK_SOURCE_GT_RENDER_EMPTY))
        if np.any(slots[~replacement] != -1) or np.any(slots[replacement] < 0) or len(np.unique(slots[replacement])) != int(replacement.sum()) or set(slots[replacement].tolist()) != set(range(len(payloads))):
            raise ValueError(f"Encoded {kind} replacement slots are inconsistent with provenance")
        if any(payload.ndim != 1 or payload.dtype != np.dtype("uint8") for payload in payloads):
            raise TypeError(f"Encoded {kind} replacement payloads must be one-dimensional uint8 arrays")
        normalized[kind] = {"provenance": provenance, "replacement_slot": slots, "replacement_png": payloads}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    payload_dtype = h5py.vlen_dtype(np.dtype("uint8"))
    string_dtype = h5py.string_dtype(encoding="utf-8")
    metadata_value = {"raw_mask_identities": dict(raw_mask_identities), **dict(metadata or {})}
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = EFFECTIVE_MASK_SCHEMA
            handle.attrs["revision"] = EFFECTIVE_MASK_REVISION
            handle.attrs["complete"] = False
            handle.attrs["sequence"] = str(sequence)
            handle.attrs["camera_id"] = int(camera_id)
            handle.attrs["camera_name"] = str(camera_name)
            handle.create_dataset("frames", data=np.asarray(frames, dtype=object), dtype=string_dtype)
            handle.create_dataset("metadata_json", data=json.dumps(metadata_value, sort_keys=True, separators=(",", ":")), dtype=string_dtype)
            for kind in MASK_KINDS:
                group = handle.create_group(kind)
                provenance = normalized[kind]["provenance"]
                slots = normalized[kind]["replacement_slot"]
                payloads = normalized[kind]["replacement_png"]
                digest = hashlib.sha256()
                digest.update(provenance.tobytes())
                digest.update(slots.tobytes())
                for payload in payloads:
                    digest.update(payload.tobytes())
                handle.attrs[f"{kind}_logical_sha256"] = digest.hexdigest()
                group.create_dataset("provenance", data=provenance, compression="lzf", shuffle=True)
                group.create_dataset("replacement_slot", data=slots, compression="lzf", shuffle=True)
                payload_dataset = group.create_dataset("replacement_png", shape=(len(payloads),), dtype=payload_dtype)
                for slot, payload in enumerate(payloads):
                    payload_dataset[slot] = payload
            handle.attrs.modify("complete", True)
            handle.flush()
            validate_effective_mask_handle(handle, expected_sequence=sequence, expected_camera_id=camera_id, expected_frames=frames, expected_source_identities=raw_mask_identities)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_effective_masks(path: str | Path, *, sequence: str, camera_id: int, camera_name: str, frames: Sequence[str], raw_empty: Mapping[str, np.ndarray], pose_valid: Mapping[str, np.ndarray], replacements: Mapping[str, Mapping[int, np.ndarray]], raw_mask_identities: Mapping[str, Mapping[str, Any]], metadata: Mapping[str, Any] | None = None, overwrite: bool = False) -> Path:
    frames = [str(frame) for frame in frames]
    frame_count = len(frames)
    if frame_count == 0:
        raise ValueError("Effective-mask H5 requires at least one frame")
    for kind in MASK_KINDS:
        empty = np.asarray(raw_empty[kind], dtype=bool)
        valid = np.asarray(pose_valid[kind], dtype=bool)
        if empty.shape != (frame_count,):
            raise ValueError(f"raw_empty[{kind!r}] must have shape {(frame_count,)}, got {empty.shape}")
        if valid.shape != (frame_count,):
            raise ValueError(f"pose_valid[{kind!r}] must have shape {(frame_count,)}, got {valid.shape}")
        expected_replacements = set(np.flatnonzero(empty & valid).tolist())
        if set(int(index) for index in replacements[kind]) != expected_replacements:
            raise ValueError(f"Replacement indices for {kind} do not exactly match empty raw-mask frames with valid poses")
    encoded = {}
    for kind in MASK_KINDS:
        empty = np.asarray(raw_empty[kind], dtype=bool)
        valid = np.asarray(pose_valid[kind], dtype=bool)
        indices = np.flatnonzero(empty & valid).astype(np.int32)
        provenance = np.full(frame_count, MASK_SOURCE_RAW, dtype=np.uint8)
        provenance[empty & ~valid] = MASK_SOURCE_RAW_EMPTY_INVALID_POSE
        slots = np.full(frame_count, -1, dtype=np.int32)
        payloads = []
        for slot, index in enumerate(indices):
            replacement = np.asarray(replacements[kind][int(index)], dtype=bool)
            provenance[index] = MASK_SOURCE_GT_RENDER if np.any(replacement) else MASK_SOURCE_GT_RENDER_EMPTY
            slots[index] = slot
            payloads.append(encode_binary_mask(replacement))
        encoded[kind] = {"provenance": provenance, "replacement_slot": slots, "replacement_png": payloads}
    return write_encoded_effective_masks(path, sequence=sequence, camera_id=camera_id, camera_name=camera_name, frames=frames, encoded=encoded, raw_mask_identities=raw_mask_identities, metadata=metadata, overwrite=overwrite)
