from __future__ import annotations

import os
import re
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from lib_mhr.human_texture import MHR_PART_TEXTURE_REVISION
from lib_mhr.object_texture import MHR_OBJECT_RENDER_MATERIAL_REVISION
from prep.mhr_foundationpose_training_tiers import FOUNDATIONPOSE_TRAINING_TIER_REVISION
from prep.mhr_geometry_crop import MHR_GEOMETRY_CROP_REVISION
from render_h5_codec import RECORD_CODEC_ATTR, create_pickled_dataset, decode_pickled_payload, load_pickled_dataset


MHR_RENDER_H5_FORMAT = "cari4d_mhr_render_v1"
MHR_RENDER_RECORD_MODE_LEGACY = "legacy_perturb_0"
MHR_RENDER_RECORD_MODE_TIER_ONLY = "foundationpose_training_tiers_only"
RENDER_KEY_PATTERN = re.compile(r"^(?P<frame>.+)_k(?P<camera>\d+)_(?P<kind>input|perturb_0|tier_[123])$")


def _file_identity(path: str | Path) -> tuple[int, int, int, int]:
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def packed_source_identity(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    identity = {"path": str(resolved), "device": int(stat.st_dev), "inode": int(stat.st_ino), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    if h5py.is_hdf5(resolved):
        with h5py.File(resolved, "r") as handle:
            identity["schema"] = str(handle.attrs.get("schema", ""))
            identity["sequence"] = str(handle.attrs.get("sequence", ""))
            if "metadata_json" in handle:
                encoded_metadata = handle["metadata_json"][()]
                encoded_metadata = bytes(encoded_metadata).decode("utf-8") if isinstance(encoded_metadata, (bytes, np.bytes_, np.void)) else str(encoded_metadata)
                metadata = json.loads(encoded_metadata)
            else:
                metadata = {}
            identity["foundationpose_training_tier_revision"] = metadata.get("foundationpose_training_tier_revision")
    return identity


def validate_render_h5(path: str | Path, *, seq_name: str, expected_cameras: Sequence[int] | None = None, expected_frame_count: int | None = None, expected_frames: Sequence[str] | None = None, validation_workers: int = 1, validation_batch_size: int = 32, decode_records: bool = True, expected_packed_source: str | Path | dict[str, object] | None = None) -> dict[str, object]:
    if validation_workers <= 0 or validation_batch_size <= 0:
        raise ValueError(f"validation_workers and validation_batch_size must be positive, got {validation_workers}, {validation_batch_size}")
    path = Path(path).resolve()
    expected_frames = None if expected_frames is None else [str(frame) for frame in expected_frames]
    if expected_frames is not None and len(expected_frames) != len(set(expected_frames)):
        raise ValueError("Expected MHR render frame names contain duplicates")
    if expected_frames is not None and expected_frame_count is not None and len(expected_frames) != int(expected_frame_count):
        raise ValueError(f"Expected MHR render frame count differs: names={len(expected_frames)}, count={expected_frame_count}")
    executor = ThreadPoolExecutor(max_workers=validation_workers) if decode_records and validation_workers > 1 else None
    try:
        with h5py.File(path, "r") as handle:
            file_format = handle.attrs.get("format")
            if isinstance(file_format, (bytes, np.bytes_)):
                file_format = bytes(file_format).decode("utf-8")
            if file_format != MHR_RENDER_H5_FORMAT:
                raise ValueError(f"MHR render H5 format must be {MHR_RENDER_H5_FORMAT!r}, got {file_format!r}")
            if not bool(handle.attrs.get("complete", False)):
                raise ValueError(f"MHR render H5 is incomplete: {path}")
            metadata_key = f"{seq_name}_w2c"
            if metadata_key not in handle:
                raise KeyError(f"MHR render H5 is missing {metadata_key}")
            metadata = load_pickled_dataset(handle[metadata_key])
            expected_source_identity = packed_source_identity(expected_packed_source) if isinstance(expected_packed_source, (str, Path)) else expected_packed_source
            actual_source_identity = metadata.get("packed_source_identity")
            if expected_source_identity is not None and actual_source_identity != expected_source_identity:
                raise ValueError(f"MHR render H5 packed-source identity differs: expected={expected_source_identity}, actual={actual_source_identity}")
            if expected_source_identity is not None:
                if metadata.get("mhr_human_render_material_revision") != MHR_PART_TEXTURE_REVISION:
                    raise ValueError(f"MHR render H5 has stale human material revision {metadata.get('mhr_human_render_material_revision')!r}")
                if metadata.get("mhr_object_render_material_revision") != MHR_OBJECT_RENDER_MATERIAL_REVISION:
                    raise ValueError(f"MHR render H5 has stale object material revision {metadata.get('mhr_object_render_material_revision')!r}")
                if metadata.get("mhr_geometry_crop_revision") != MHR_GEOMETRY_CROP_REVISION:
                    raise ValueError(f"MHR render H5 has stale geometry-crop revision {metadata.get('mhr_geometry_crop_revision')!r}")
            cameras = [int(value) for value in np.asarray(metadata["kids"]).tolist()]
            if len(cameras) != len(set(cameras)):
                raise ValueError(f"MHR render H5 contains duplicate cameras: {cameras}")
            if expected_cameras is not None and cameras != [int(camera) for camera in expected_cameras]:
                raise ValueError(f"MHR render H5 cameras differ: expected={list(expected_cameras)}, actual={cameras}")
            tier_revision = metadata.get("foundationpose_training_tier_revision")
            if tier_revision is not None and tier_revision != FOUNDATIONPOSE_TRAINING_TIER_REVISION:
                raise ValueError(f"MHR render H5 has stale FoundationPose training-tier revision {tier_revision!r}")
            tier_frames = [str(frame) for frame in metadata.get("foundationpose_training_tier_frames", [])]
            tier_valid = np.asarray(metadata.get("foundationpose_training_tier_valid", []))
            if tier_revision is not None:
                if not tier_frames or len(tier_frames) != len(set(tier_frames)):
                    raise ValueError("MHR render H5 training-tier frame metadata must be nonempty and unique")
                if tier_valid.shape != (len(tier_frames), len(cameras), 3) or not np.isin(tier_valid, (0, 1)).all():
                    raise ValueError(f"MHR render H5 training-tier validity expected shape ({len(tier_frames)},{len(cameras)},3), got {tier_valid.shape}")
                tier_valid = tier_valid.astype(bool)
            record_mode = metadata.get("object_initialization_record_mode")
            expected_record_mode = MHR_RENDER_RECORD_MODE_TIER_ONLY if tier_revision is not None else MHR_RENDER_RECORD_MODE_LEGACY
            if tier_revision is not None and record_mode != expected_record_mode:
                raise ValueError(f"MHR render H5 object initialization record mode must be {expected_record_mode!r}, got {record_mode!r}")
            if tier_revision is None and record_mode not in (None, expected_record_mode):
                raise ValueError(f"MHR render H5 object initialization record mode must be absent or {expected_record_mode!r}, got {record_mode!r}")
            record_mode = expected_record_mode
            records = {camera: {"input": set(), "perturb_0": set(), "tier_1": set(), "tier_2": set(), "tier_3": set()} for camera in cameras}
            pending = []
            for key in handle:
                if key == metadata_key:
                    continue
                match = RENDER_KEY_PATTERN.match(key)
                if match is None:
                    raise ValueError(f"Unexpected MHR render H5 key: {key}")
                camera = int(match.group("camera"))
                if camera not in records:
                    raise ValueError(f"MHR render key {key} refers to camera {camera}, absent from metadata cameras={cameras}")
                frame_key = match.group("frame")
                prefix = f"{seq_name}+"
                if not frame_key.startswith(prefix):
                    raise ValueError(f"MHR render key {key} does not belong to sequence {seq_name}")
                records[camera][match.group("kind")].add(frame_key[len(prefix):])
                if decode_records:
                    node = handle[key]
                    pending.append((bytes(node[()]), node.attrs.get(RECORD_CODEC_ATTR)))
                    if len(pending) == validation_batch_size:
                        list(executor.map(lambda values: decode_pickled_payload(*values), pending)) if executor is not None else [decode_pickled_payload(*values) for values in pending]
                        pending.clear()
            if decode_records and pending:
                list(executor.map(lambda values: decode_pickled_payload(*values), pending)) if executor is not None else [decode_pickled_payload(*values) for values in pending]
            frame_counts = {}
            for camera in cameras:
                inputs = records[camera]["input"]
                renders = records[camera]["perturb_0"]
                if expected_frame_count is not None and len(inputs) != int(expected_frame_count):
                    raise ValueError(f"MHR render H5 frame count differs for camera {camera}: expected={expected_frame_count}, actual={len(inputs)}")
                if expected_frames is not None and inputs != set(expected_frames):
                    raise ValueError(f"MHR render H5 frames differ for camera {camera}: expected={len(expected_frames)}, actual={len(inputs)}")
                tier_records_present = any(records[camera][f"tier_{tier}"] for tier in (1, 2, 3))
                if tier_records_present and tier_revision is None:
                    raise ValueError(f"MHR render H5 camera {camera} has tier records without training-tier metadata")
                if tier_revision is not None:
                    if renders:
                        raise ValueError(f"MHR render H5 tier-only camera {camera} contains forbidden perturb_0 records")
                    camera_index = cameras.index(camera)
                    if inputs != set(tier_frames):
                        raise ValueError(f"MHR render H5 input frames differ from training-tier metadata for camera {camera}")
                    for tier in (1, 2, 3):
                        expected_tier_frames = {tier_frames[index] for index in np.flatnonzero(tier_valid[:, camera_index, tier - 1])}
                        if records[camera][f"tier_{tier}"] != expected_tier_frames:
                            raise ValueError(f"MHR render H5 tier {tier} records differ for camera {camera}: expected={len(expected_tier_frames)}, actual={len(records[camera][f'tier_{tier}'])}")
                else:
                    if inputs != renders:
                        raise ValueError(f"MHR render H5 input/render frame sets differ for camera {camera}: input={len(inputs)} render={len(renders)}")
                    if tier_records_present:
                        raise ValueError(f"MHR render H5 legacy camera {camera} contains unexpected training-tier records")
                frame_counts[camera] = len(inputs)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return {"path": str(path), "format": MHR_RENDER_H5_FORMAT, "sequence": seq_name, "cameras": cameras, "frame_counts": frame_counts, "foundationpose_training_tier_revision": tier_revision, "object_initialization_record_mode": record_mode, "mhr_geometry_crop_revision": metadata.get("mhr_geometry_crop_revision"), "packed_source_identity": actual_source_identity, "records_decoded": bool(decode_records)}


def merge_render_h5_shards(output_path: str | Path, shard_paths: Sequence[str | Path], *, seq_name: str, expected_cameras: Sequence[int], expected_frame_count: int | None = None, expected_frames: Sequence[str] | None = None, redo: bool = False, remove_shards: bool = False, validation_workers: int = 8, validation_batch_size: int = 32, decode_records: bool = True, expected_packed_source: str | Path | dict[str, object] | None = None) -> dict[str, object]:
    output_path = Path(output_path).resolve()
    shards = [Path(path).resolve() for path in shard_paths]
    expected_cameras = [int(camera) for camera in expected_cameras]
    if not shards:
        raise ValueError("At least one MHR render H5 shard is required")
    if len(set(shards)) != len(shards):
        raise ValueError("MHR render H5 shard paths must be unique")
    if output_path.exists() and not redo:
        return validate_render_h5(output_path, seq_name=seq_name, expected_cameras=expected_cameras, expected_frame_count=expected_frame_count, expected_frames=expected_frames, decode_records=decode_records, expected_packed_source=expected_packed_source)
    if validation_workers <= 0 or validation_batch_size <= 0:
        raise ValueError(f"validation_workers and validation_batch_size must be positive, got {validation_workers}, {validation_batch_size}")
    shard_workers = min(len(shards), validation_workers)
    workers_per_shard = max(1, validation_workers // shard_workers)
    with ThreadPoolExecutor(max_workers=shard_workers) as executor:
        reports = list(executor.map(lambda path: validate_render_h5(path, seq_name=seq_name, expected_frame_count=expected_frame_count, expected_frames=expected_frames, validation_workers=workers_per_shard, validation_batch_size=validation_batch_size, decode_records=decode_records, expected_packed_source=expected_packed_source), shards))
    shard_cameras = [camera for report in reports for camera in report["cameras"]]
    if len(shard_cameras) != len(set(shard_cameras)):
        raise ValueError(f"MHR render H5 shards contain duplicate cameras: {shard_cameras}")
    if sorted(shard_cameras) != sorted(expected_cameras):
        raise ValueError(f"MHR render H5 shard cameras differ: expected={expected_cameras}, actual={shard_cameras}")
    metadata_by_camera = {}
    metadata_key = f"{seq_name}_w2c"
    for shard in shards:
        with h5py.File(shard, "r") as source:
            metadata = load_pickled_dataset(source[metadata_key])
            for index, camera in enumerate(np.asarray(metadata["kids"]).tolist()):
                camera_metadata = {"rot": np.asarray(metadata["rot"])[index], "trans": np.asarray(metadata["trans"])[index], "mesh_diameter": float(metadata["mesh_diameter"]), "trans_normalizer": np.asarray(metadata["trans_normalizer"]), "rot_normalizer": float(metadata["rot_normalizer"]), "object_initialization_record_mode": metadata.get("object_initialization_record_mode", MHR_RENDER_RECORD_MODE_TIER_ONLY if metadata.get("foundationpose_training_tier_revision") is not None else MHR_RENDER_RECORD_MODE_LEGACY), "mhr_human_render_material_revision": metadata.get("mhr_human_render_material_revision"), "mhr_object_render_material_revision": metadata.get("mhr_object_render_material_revision"), "mhr_geometry_crop_revision": metadata.get("mhr_geometry_crop_revision")}
                if metadata.get("foundationpose_training_tier_revision") is not None:
                    camera_metadata.update({"foundationpose_training_tier_revision": metadata["foundationpose_training_tier_revision"], "foundationpose_training_tier_frames": [str(frame) for frame in metadata["foundationpose_training_tier_frames"]], "foundationpose_training_tier_valid": np.asarray(metadata["foundationpose_training_tier_valid"], dtype=bool)[:, index]})
                camera_metadata["packed_source_identity"] = metadata.get("packed_source_identity")
                metadata_by_camera[int(camera)] = camera_metadata
    reference = metadata_by_camera[expected_cameras[0]]
    for camera in expected_cameras[1:]:
        current = metadata_by_camera[camera]
        if not np.array_equal(current["trans_normalizer"], reference["trans_normalizer"]) or current["rot_normalizer"] != reference["rot_normalizer"]:
            raise ValueError(f"MHR render normalization metadata differs for camera {camera}")
        if current.get("foundationpose_training_tier_revision") != reference.get("foundationpose_training_tier_revision") or current.get("foundationpose_training_tier_frames") != reference.get("foundationpose_training_tier_frames"):
            raise ValueError(f"MHR render FoundationPose training-tier metadata differs for camera {camera}")
        if current["object_initialization_record_mode"] != reference["object_initialization_record_mode"]:
            raise ValueError(f"MHR render object initialization record mode differs for camera {camera}")
        if current.get("packed_source_identity") != reference.get("packed_source_identity"):
            raise ValueError(f"MHR render packed-source identity differs for camera {camera}")
        if current.get("mhr_human_render_material_revision") != reference.get("mhr_human_render_material_revision") or current.get("mhr_object_render_material_revision") != reference.get("mhr_object_render_material_revision"):
            raise ValueError(f"MHR render material revisions differ for camera {camera}")
        if current.get("mhr_geometry_crop_revision") != reference.get("mhr_geometry_crop_revision"):
            raise ValueError(f"MHR render geometry-crop revisions differ for camera {camera}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.merge-{os.getpid()}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        with h5py.File(temporary_path, "w") as target:
            target.attrs["format"] = MHR_RENDER_H5_FORMAT
            target.attrs["complete"] = False
            for shard in shards:
                with h5py.File(shard, "r") as source:
                    for key in source:
                        if key == metadata_key:
                            continue
                        if key in target:
                            raise ValueError(f"Duplicate MHR render record across shards: {key}")
                        source.copy(source[key], target, name=key)
            merged_metadata = {"kids": np.asarray(expected_cameras, dtype=np.int16), "rot": np.stack([metadata_by_camera[camera]["rot"] for camera in expected_cameras]), "trans": np.stack([metadata_by_camera[camera]["trans"] for camera in expected_cameras]), "mesh_diameter": reference["mesh_diameter"], "trans_normalizer": reference["trans_normalizer"], "rot_normalizer": reference["rot_normalizer"], "object_initialization_record_mode": reference["object_initialization_record_mode"], "mhr_human_render_material_revision": reference["mhr_human_render_material_revision"], "mhr_object_render_material_revision": reference["mhr_object_render_material_revision"], "mhr_geometry_crop_revision": reference["mhr_geometry_crop_revision"]}
            if reference.get("packed_source_identity") is not None:
                merged_metadata["packed_source_identity"] = reference["packed_source_identity"]
            if reference.get("foundationpose_training_tier_revision") is not None:
                merged_metadata.update({"foundationpose_training_tier_revision": reference["foundationpose_training_tier_revision"], "foundationpose_training_tier_frames": reference["foundationpose_training_tier_frames"], "foundationpose_training_tier_valid": np.stack([metadata_by_camera[camera]["foundationpose_training_tier_valid"] for camera in expected_cameras], axis=1)})
            create_pickled_dataset(target, metadata_key, merged_metadata)
            target.attrs.modify("complete", True)
            target.flush()
        report = validate_render_h5(temporary_path, seq_name=seq_name, expected_cameras=expected_cameras, expected_frame_count=expected_frame_count, expected_frames=expected_frames, validation_workers=validation_workers, validation_batch_size=validation_batch_size, decode_records=decode_records, expected_packed_source=expected_packed_source)
        temporary_identity = _file_identity(temporary_path)
        os.replace(temporary_path, output_path)
        if _file_identity(output_path) != temporary_identity:
            raise RuntimeError(f"Atomic MHR render H5 publish changed file identity: {output_path}")
        report = {**report, "path": str(output_path)}
        if remove_shards:
            for shard in shards:
                shard.unlink()
        return report
    finally:
        temporary_path.unlink(missing_ok=True)
