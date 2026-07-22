# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Commit and validate crash-safe SAM2 mask generations."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


RUN_GENERATION_FILENAME = "run_generation.json"
RUN_GENERATION_SCHEMA = "v2d.sam2.video-to-masks-generation/v1"
_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {"size_bytes": stat.st_size, "sha256": _sha256(path)}


def _named_files_identity(paths: dict[str, Path]) -> dict:
    files: dict[str, dict] = {}
    digest = hashlib.sha256()
    total_size = 0
    for name, path in sorted(paths.items()):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"SAM2 generation artifact must be a regular file: {path}"
            )
        identity = _file_identity(path)
        encoded_name = name.encode("utf-8")
        size = int(identity["size_bytes"])
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[name] = identity
        total_size += size
    return {
        "files": files,
        "file_count": len(files),
        "size_bytes": total_size,
        "aggregate_sha256": digest.hexdigest(),
    }


def _source_identity(path: Path) -> dict:
    if path.is_file():
        return {"kind": "file", "artifact": _file_identity(path)}
    if not path.is_dir():
        raise FileNotFoundError(path)
    frames = {
        child.name: child
        for child in path.iterdir()
        if child.suffix.lower() in _IMAGE_SUFFIXES
    }
    if not frames:
        raise RuntimeError(f"SAM2 image source directory contains no frames: {path}")
    return {"kind": "image_directory", **_named_files_identity(frames)}


def build_static_identity(
    video_path: str,
    prompts_path: str,
    weights_dir: str,
    prompt_records: list[dict],
    image_id: str,
) -> dict:
    """Fingerprint every input that affects a strict single-video generation."""

    if not _IMAGE_ID_PATTERN.fullmatch(image_id):
        raise ValueError(
            "image_id must be an immutable sha256:<64 lowercase hex> Docker ID"
        )
    video = Path(video_path)
    prompts = Path(prompts_path)
    weights = Path(weights_dir)
    checkpoint_name = os.environ.get("CHECKPOINT_FILE", "sam2.1_hiera_large.pt")
    config_file = os.environ.get("CONFIG_FILE", "configs/sam2.1/sam2.1_hiera_l.yaml")
    weights_config = weights / config_file
    prompt_root = prompts.resolve().parent
    prompt_masks = []
    for index, record in enumerate(prompt_records):
        mask_path = record.get("mask_path")
        if not mask_path:
            continue
        path = Path(mask_path)
        if not path.is_absolute():
            path = prompt_root / path
        prompt_masks.append(
            {
                "prompt_index": index,
                "object_id": int(record["object_id"]),
                "frame_index": int(record["frame_index"]),
                "artifact": _file_identity(path),
            }
        )
    return {
        "execution_environment": {"container_image_id": image_id},
        "implementation_sources": {
            name: _file_identity(Path(__file__).with_name(name))
            for name in ("generation.py", "sam2_utils.py", "video_to_masks.py")
        },
        "video": _source_identity(video),
        "prompts_json": _file_identity(prompts),
        "prompt_masks": prompt_masks,
        "checkpoint": {
            "filename": checkpoint_name,
            "artifact": _file_identity(weights / checkpoint_name),
        },
        "weights_config": {
            "filename": config_file,
            # Some images resolve the pinned config from their installed SAM2
            # package instead of the weights mount. The immutable image ID
            # covers that case; if the weights search path supplies the config,
            # its bytes are independently committed here.
            "artifact": (
                _file_identity(weights_config) if weights_config.is_file() else None
            ),
        },
        "parameters": {
            "config_file": config_file,
            "mask_extension": "",
            "frame_index_origin": 0,
            "mask_encoding": "8-bit PNG, foreground=255",
        },
    }


def _generation_id(
    static_identity: dict, object_ids: list[int], frame_count: int
) -> str:
    payload = {
        "static_identity": static_identity,
        "expected_object_ids": object_ids,
        "expected_frame_count": frame_count,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _outputs_identity(
    output_dir: Path,
    object_ids: list[int],
    frame_count: int,
    *,
    manifest_expected: bool,
) -> dict:
    expected_top_level = {str(object_id) for object_id in object_ids}
    if manifest_expected:
        expected_top_level.add(RUN_GENERATION_FILENAME)
    actual_top_level = {path.name for path in output_dir.iterdir()}
    if actual_top_level != expected_top_level:
        raise RuntimeError(
            "SAM2 output directory does not contain exactly the committed "
            "object directories and generation manifest"
        )

    expected_frames = [f"{index:06d}.png" for index in range(frame_count)]
    expected_frame_set = set(expected_frames)
    objects: dict[str, dict] = {}
    for object_id in object_ids:
        object_dir = output_dir / str(object_id)
        if object_dir.is_symlink() or not object_dir.is_dir():
            raise RuntimeError(
                f"SAM2 object output is not a regular directory: {object_dir}"
            )
        actual_frames = {path.name for path in object_dir.iterdir()}
        if actual_frames != expected_frame_set:
            raise RuntimeError(
                f"SAM2 object {object_id} does not contain exactly {frame_count} frames"
            )
        objects[str(object_id)] = _named_files_identity(
            {name: object_dir / name for name in expected_frames}
        )
    return {"objects": objects}


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def commit_generation(
    output_dir: Path,
    static_identity: dict,
    object_ids: list[int],
    frame_count: int,
) -> dict:
    """Write the complete marker last, after validating and hashing all masks."""

    object_ids = sorted(set(int(value) for value in object_ids))
    if not object_ids or frame_count <= 0:
        raise RuntimeError("SAM2 cannot commit an empty object or frame generation")
    outputs = _outputs_identity(
        output_dir, object_ids, frame_count, manifest_expected=False
    )
    manifest = {
        "schema_version": RUN_GENERATION_SCHEMA,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "static_identity": static_identity,
        "expected": {"object_ids": object_ids, "frame_count": frame_count},
        "generation_id": _generation_id(static_identity, object_ids, frame_count),
        "outputs": outputs,
    }
    _atomic_json(output_dir / RUN_GENERATION_FILENAME, manifest)
    return manifest


def validate_generation(output_dir: Path, current_static_identity: dict) -> dict:
    """Validate an existing complete generation against current inputs and bytes."""

    manifest_path = output_dir / RUN_GENERATION_FILENAME
    if not manifest_path.is_file():
        raise FileExistsError(
            f"Existing SAM2 output has no {RUN_GENERATION_FILENAME}; refusing to "
            "skip, overwrite, or mix mask generations"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid SAM2 generation manifest: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != RUN_GENERATION_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise RuntimeError("SAM2 generation manifest is not a complete v1 commit")
    if manifest.get("static_identity") != current_static_identity:
        raise RuntimeError(
            "Existing SAM2 masks belong to different video/prompt/weight/image inputs; "
            "refusing resume"
        )
    expected = manifest.get("expected")
    if not isinstance(expected, dict):
        raise RuntimeError("SAM2 generation manifest omits expected output geometry")
    object_ids = expected.get("object_ids")
    frame_count = expected.get("frame_count")
    if (
        not isinstance(object_ids, list)
        or not object_ids
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in object_ids
        )
        or object_ids != sorted(set(object_ids))
        or isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
    ):
        raise RuntimeError("SAM2 generation manifest has invalid expected outputs")
    if manifest.get("generation_id") != _generation_id(
        current_static_identity, object_ids, frame_count
    ):
        raise RuntimeError("SAM2 generation ID is invalid")
    outputs = _outputs_identity(
        output_dir, object_ids, frame_count, manifest_expected=True
    )
    if manifest.get("outputs") != outputs:
        raise RuntimeError("SAM2 output hashes no longer match the generation manifest")
    return manifest
