"""Build an RGB-only tool/target occluder bundle from MoGe-2 and SAM2.

The producer intentionally estimates depth on the original RGB frames, then
keeps depth only where RGB-prompted SAM2 sees the two task objects.  Pixels in
the effective E2FGVI arm-removal region are excluded: their original RGB depth
belongs to the removed hand/sleeve and must not hide the replacement robot.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any
import uuid

import cv2
import numpy as np

from .contracts import ContractError, VideoGeometry
from .occluder_depth import (
    DEPTH_SEMANTICS,
    OCCLUDER_ARTIFACT_NAMES,
    OCCLUDER_DEPTH_PROVENANCE_SCHEMA,
    OCCLUDER_DEPTH_SCHEMA,
    OCCLUDER_METADATA_NAME,
    validate_occluder_depth_bundle,
)
from .robot_renderer.provenance import file_record
from .video_io import probe_video


INPUT_MANIFEST_SCHEMA = "v2d.inpainting.rgb-occluder-input-manifest/v1"
MOGE_RUN_GENERATION_SCHEMA = "v2d.moge.video-to-depth-generation/v1"
MOGE_REPOSITORY = "Ruicheng/moge-2-vitl-normal"
MOGE_REVISION = "b135031bae30b5ac2ae141a0e68717795ce38340"
MOGE_SOURCE_COMMIT = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
MOGE_MODEL_SHA256 = "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01"
SAM2_SOURCE_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_CHECKPOINT_SHA256 = (
    "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
)
SAM2_RUN_GENERATION_SCHEMA = "v2d.sam2.video-to-masks-generation/v1"
SAM2_PROMPTS_SCHEMA = "v2d.inpainting.sam2-prompts/v1"
SAM2_CHECKPOINT_FILENAME = "sam2.1_hiera_large.pt"
SAM2_CONFIG_FILE = "configs/sam2.1/sam2.1_hiera_l.yaml"
# The SAM2 generation schema predates an explicit source-revision field.  These
# are the exact producer sources in the pinned SAM2 image/source commit above;
# accepting that commit therefore requires all three byte identities.
SAM2_IMPLEMENTATION_SOURCES = {
    "generation.py": {
        "size_bytes": 10566,
        "sha256": "88b2ccf2b1bb1fa4aa302d3238a735860bcafc3ceba906ca7b5e7b939326e06a",
    },
    "sam2_utils.py": {
        "size_bytes": 57619,
        "sha256": "603a9cea368098fd50137ad6bf4c34c81289c042d15ad1bc4a99992f744b2408",
    },
    "video_to_masks.py": {
        "size_bytes": 8992,
        "sha256": "e49f8479518ef601adfe8da78c5347d2ed266a4853e9c07e47618efe4cc4a6c6",
    },
}
SAM2_GENERATION_PARAMETERS = {
    "config_file": SAM2_CONFIG_FILE,
    "frame_index_origin": 0,
    "mask_encoding": "8-bit PNG, foreground=255",
    "mask_extension": "",
}
IMPLEMENTATION_FILES = (
    "inpainting/contracts.py",
    "inpainting/occluder_depth.py",
    "inpainting/rgb_estimated_occluder.py",
    "inpainting/robot_renderer/provenance.py",
    "inpainting/video_io.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _expected_numbered_files(
    directory: Path, *, suffix: str, frame_count: int, label: str
) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"{label}: {directory}")
    expected = [directory / f"{index:06d}{suffix}" for index in range(frame_count)]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} is missing frames: {missing[:5]}")
    actual = sorted(directory.glob(f"*{suffix}"))
    if actual != expected:
        raise ContractError(
            f"{label} must contain exactly {frame_count} numbered {suffix} files"
        )
    return expected


def _file_records(paths: list[Path]) -> list[dict[str, Any]]:
    return [file_record(path) for path in paths]


def _moge_artifact_identity(path: Path) -> dict[str, Any]:
    record = file_record(path)
    return {"size_bytes": record["bytes"], "sha256": record["sha256"]}


def _moge_directory_identity(paths: list[Path], *, directory: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files: dict[str, dict[str, Any]] = {}
    total_size = 0
    for path in paths:
        identity = _moge_artifact_identity(path)
        name = path.name.encode("utf-8")
        size = int(identity["size_bytes"])
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[path.name] = identity
        total_size += size
    return {
        "directory": str(directory.resolve()),
        "file_count": len(paths),
        "size_bytes": total_size,
        "aggregate_sha256": digest.hexdigest(),
        "files": files,
    }


def _moge_static_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    def artifact(value: object, *, label: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ContractError(f"{label} must be an artifact object")
        size = value.get("size_bytes")
        digest = value.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ContractError(f"{label} size_bytes must be non-negative")
        if re.fullmatch(r"[0-9a-f]{64}", digest or "") is None:
            raise ContractError(f"{label} sha256 is invalid")
        return {"size_bytes": size, "sha256": digest}

    try:
        sources = manifest["sources"]
        model = manifest["model"]
        implementation = manifest["implementation_sources"]
        input_intrinsics = sources["input_intrinsics"]
        if input_intrinsics is not None:
            input_intrinsics = artifact(input_intrinsics, label="MoGe input intrinsics")
        if not isinstance(implementation, dict) or not implementation:
            raise ContractError("MoGe implementation_sources must be non-empty")
        return {
            "execution_environment": manifest["execution_environment"],
            "source_revisions": manifest["source_revisions"],
            "sources": {
                "video": artifact(sources["video"], label="MoGe source video"),
                "input_intrinsics": input_intrinsics,
            },
            "model": {
                "checkpoint": artifact(model["checkpoint"], label="MoGe checkpoint")
            },
            "implementation_sources": {
                name: artifact(value, label=f"MoGe implementation {name}")
                for name, value in sorted(implementation.items())
            },
            "parameters": manifest["parameters"],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"MoGe generation provenance is incomplete: {exc}") from exc


def _moge_outputs_identity(outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {key: value for key, value in output.items() if key != "directory"}
        for name, output in sorted(outputs.items())
    }


def _moge_generation_id(
    static_identity: dict[str, Any], outputs: dict[str, Any]
) -> str:
    encoded = json.dumps(
        {
            "static_identity": static_identity,
            "outputs": _moge_outputs_identity(outputs),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_rgb_only_moge_generation(
    manifest_path: str | Path,
    *,
    source_video: Path,
    depth_files: list[Path],
    intrinsics_files: list[Path],
    validity_files: list[Path],
    moge_checkpoint: Path,
    moge_image_id: str,
) -> dict[str, Any]:
    """Prove that the supplied MoGe rasters came from RGB without a GT-K prior."""

    manifest_path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Could not read MoGe generation: {manifest_path}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MOGE_RUN_GENERATION_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise ContractError("MoGe generation is not a complete v1 commit")

    static_identity = _moge_static_identity(manifest)
    if manifest.get("static_identity") != static_identity:
        raise ContractError("MoGe top-level provenance contradicts static_identity")
    expected_revisions = {
        "moge_repository": MOGE_REPOSITORY,
        "moge_huggingface_revision": MOGE_REVISION,
        "moge_source_commit": MOGE_SOURCE_COMMIT,
    }
    if static_identity["source_revisions"] != expected_revisions:
        raise ContractError("MoGe generation does not use the pinned source revisions")
    if static_identity["execution_environment"] != {
        "container_image_id": moge_image_id
    }:
        raise ContractError("MoGe generation container image ID does not match")
    if static_identity["sources"]["video"] != _moge_artifact_identity(source_video):
        raise ContractError("MoGe generation source video does not match RGB input")
    if static_identity["model"]["checkpoint"] != _moge_artifact_identity(
        moge_checkpoint
    ):
        raise ContractError("MoGe generation checkpoint does not match")

    parameters = static_identity["parameters"]
    if not isinstance(parameters, dict):
        raise ContractError("MoGe generation parameters must be an object")
    if (
        static_identity["sources"]["input_intrinsics"] is not None
        or parameters.get("input_intrinsics_path") is not None
        or parameters.get("intrinsics_mode") != "estimated_from_rgb"
    ):
        raise ContractError(
            "RGB-only compositing requires MoGe inference with no input "
            "intrinsics/calibration prior"
        )
    requested_outputs = parameters.get("requested_outputs")
    if not isinstance(requested_outputs, list) or not {
        "depth",
        "intrinsics",
        "mask",
    }.issubset(requested_outputs):
        raise ContractError("MoGe generation omits required output modalities")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ContractError("MoGe generation outputs must be an object")
    expected_paths = {
        "depth": depth_files,
        "intrinsics": intrinsics_files,
        "mask": validity_files,
    }
    for name, paths in expected_paths.items():
        recorded = outputs.get(name)
        if not isinstance(recorded, dict):
            raise ContractError(f"MoGe generation omits {name} output identity")
        current = _moge_directory_identity(paths, directory=paths[0].parent)
        if recorded != current:
            raise ContractError(
                f"MoGe {name} files do not match the committed generation"
            )
        allowed_entries = {path.name for path in paths}
        if manifest_path.parent == paths[0].parent:
            allowed_entries.add(manifest_path.name)
        if {path.name for path in paths[0].parent.iterdir()} != allowed_entries:
            raise ContractError(
                f"MoGe {name} directory contains uncommitted extra files"
            )

    try:
        frame_count = int(manifest["expected_frames"]["count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("MoGe generation omits expected frame count") from exc
    if frame_count != len(depth_files):
        raise ContractError("MoGe generation frame count does not match RGB video")
    if manifest["expected_frames"].get("indices") != [0, frame_count - 1]:
        raise ContractError("MoGe generation expected frame indices are invalid")
    if manifest.get("generation_id") != _moge_generation_id(static_identity, outputs):
        raise ContractError("MoGe generation ID is invalid")
    return manifest


def _sam2_generation_id(
    static_identity: dict[str, Any], object_ids: list[int], frame_count: int
) -> str:
    encoded = json.dumps(
        {
            "static_identity": static_identity,
            "expected_object_ids": object_ids,
            "expected_frame_count": frame_count,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sam2_object_output_identity(paths: list[Path]) -> dict[str, Any]:
    identity = _moge_directory_identity(paths, directory=paths[0].parent)
    return {key: value for key, value in identity.items() if key != "directory"}


def _sam2_container_prompts_identity(payload: dict[str, Any]) -> dict[str, Any]:
    # The host wrapper loads the user prompt JSON and stages it with
    # json.dump(..., indent=2), without a trailing newline.  The container
    # manifest commits those staged bytes rather than the host file formatting.
    encoded = json.dumps(payload, indent=2).encode()
    return {"size_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _load_rgb_box_prompts(
    prompts_path: Path,
    *,
    sequence_id: str,
    source_video: Path,
    geometry: VideoGeometry,
    object_ids: list[int],
) -> dict[str, Any]:
    try:
        payload = json.loads(prompts_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Could not read SAM2 prompts: {prompts_path}") from exc
    if not isinstance(payload, dict) or set(payload) != {"prompts", "metadata"}:
        raise ContractError(
            "RGB-only SAM2 prompts must contain exactly prompts and metadata"
        )

    metadata = payload["metadata"]
    expected_metadata_keys = {
        "schema_version",
        "sequence_id",
        "source_video",
        "geometry",
        "role",
        "initialization",
        "object_ids",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_metadata_keys:
        raise ContractError("SAM2 prompt metadata has an unsupported schema")
    if metadata.get("schema_version") != SAM2_PROMPTS_SCHEMA:
        raise ContractError("SAM2 prompt metadata schema is not pinned")
    if metadata.get("sequence_id") != sequence_id:
        raise ContractError("SAM2 prompts belong to a different sequence")
    prompt_source = metadata.get("source_video")
    if not isinstance(prompt_source, str) or not prompt_source:
        raise ContractError("SAM2 prompt metadata omits its RGB source video")
    prompt_source_path = Path(prompt_source)
    if not prompt_source_path.is_absolute():
        prompt_source_path = prompts_path.parent / prompt_source_path
    if prompt_source_path.resolve() != source_video:
        raise ContractError("SAM2 prompts name a different RGB source video")
    if metadata.get("geometry") != geometry.as_dict():
        raise ContractError("SAM2 prompt geometry does not match the RGB video")
    if metadata.get("role") != "rgb_only_tool_and_target_segmentation":
        raise ContractError("SAM2 prompts are not declared as RGB-only segmentation")
    if metadata.get("initialization") != "human_box_prompts_on_rgb_frame_0":
        raise ContractError("SAM2 prompts are not declared as human RGB box prompts")
    object_labels = metadata.get("object_ids")
    if (
        not isinstance(object_labels, dict)
        or set(object_labels) != {str(value) for value in object_ids}
        or any(
            not isinstance(value, str) or not value for value in object_labels.values()
        )
    ):
        raise ContractError("SAM2 prompt object metadata does not match requested IDs")

    prompts = payload["prompts"]
    if not isinstance(prompts, list) or len(prompts) != len(object_ids):
        raise ContractError("SAM2 requires exactly one RGB box prompt per object")
    expected_prompt_keys = {
        "frame_index",
        "object_id",
        "points",
        "point_labels",
        "box",
        "mask_path",
    }
    seen: set[int] = set()
    for prompt in prompts:
        if not isinstance(prompt, dict) or set(prompt) != expected_prompt_keys:
            raise ContractError("SAM2 RGB box prompt has an unsupported schema")
        frame_index = prompt.get("frame_index")
        object_id = prompt.get("object_id")
        if isinstance(frame_index, bool) or frame_index != 0:
            raise ContractError("Every SAM2 object must be prompted on RGB frame 0")
        if (
            isinstance(object_id, bool)
            or not isinstance(object_id, int)
            or object_id not in object_ids
            or object_id in seen
        ):
            raise ContractError("SAM2 prompt object IDs must be unique and requested")
        if (
            prompt.get("points") is not None
            or prompt.get("point_labels") is not None
            or prompt.get("mask_path") is not None
        ):
            raise ContractError(
                "RGB-only SAM2 initialization permits boxes only; point or mask "
                "prompts cannot establish the human-RGB-only claim"
            )
        box = prompt.get("box")
        if not isinstance(box, dict) or set(box) != {"x0", "y0", "x1", "y1"}:
            raise ContractError("Every SAM2 RGB prompt must contain one box")
        coordinates = [box[key] for key in ("x0", "y0", "x1", "y1")]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not bool(np.isfinite(value))
            for value in coordinates
        ):
            raise ContractError("SAM2 RGB box coordinates must be finite numbers")
        x0, y0, x1, y1 = (float(value) for value in coordinates)
        if not (0.0 <= x0 < x1 <= geometry.width):
            raise ContractError("SAM2 RGB box x coordinates are outside the video")
        if not (0.0 <= y0 < y1 <= geometry.height):
            raise ContractError("SAM2 RGB box y coordinates are outside the video")
        seen.add(object_id)
    if seen != set(object_ids):
        raise ContractError("SAM2 prompts do not cover every requested object")
    return payload


def validate_rgb_only_sam2_generation(
    manifest_path: str | Path,
    *,
    sequence_id: str,
    source_video: Path,
    geometry: VideoGeometry,
    prompts_path: Path,
    object_files: dict[int, list[Path]],
    sam2_checkpoint: Path,
    sam2_image_id: str,
) -> dict[str, Any]:
    """Validate a byte-exact SAM2 commit and its human RGB-box lineage."""

    manifest_path = Path(manifest_path).resolve()
    prompts_path = Path(prompts_path).resolve()
    object_ids = sorted(object_files)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", sam2_image_id) is None:
        raise ContractError("SAM2 generation image ID is not immutable")
    checkpoint_identity = _moge_artifact_identity(sam2_checkpoint)
    if checkpoint_identity["sha256"] != SAM2_CHECKPOINT_SHA256:
        raise ContractError("SAM2 generation does not use the pinned checkpoint")
    prompt_payload = _load_rgb_box_prompts(
        prompts_path,
        sequence_id=sequence_id,
        source_video=source_video,
        geometry=geometry,
        object_ids=object_ids,
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Could not read SAM2 generation: {manifest_path}") from exc
    expected_manifest_keys = {
        "schema_version",
        "state",
        "completed_at",
        "static_identity",
        "expected",
        "generation_id",
        "outputs",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_manifest_keys
        or manifest.get("schema_version") != SAM2_RUN_GENERATION_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise ContractError("SAM2 generation is not a complete canonical v1 commit")

    static_identity = {
        "execution_environment": {"container_image_id": sam2_image_id},
        "implementation_sources": SAM2_IMPLEMENTATION_SOURCES,
        "video": {
            "kind": "file",
            "artifact": _moge_artifact_identity(source_video),
        },
        "prompts_json": _sam2_container_prompts_identity(prompt_payload),
        "prompt_masks": [],
        "checkpoint": {
            "filename": SAM2_CHECKPOINT_FILENAME,
            "artifact": checkpoint_identity,
        },
        "weights_config": {
            "filename": SAM2_CONFIG_FILE,
            "artifact": None,
        },
        "parameters": SAM2_GENERATION_PARAMETERS,
    }
    if manifest.get("static_identity") != static_identity:
        raise ContractError(
            "SAM2 generation does not match the pinned video, prompts, checkpoint, "
            "container image, config, and source commit"
        )
    expected = {"object_ids": object_ids, "frame_count": geometry.frame_count}
    if manifest.get("expected") != expected:
        raise ContractError("SAM2 generation object IDs or frame count do not match")
    if manifest.get("generation_id") != _sam2_generation_id(
        static_identity, object_ids, geometry.frame_count
    ):
        raise ContractError("SAM2 generation ID is invalid")

    output_root = manifest_path.parent
    if manifest_path.name != "run_generation.json":
        raise ContractError("SAM2 generation must use run_generation.json")
    expected_top_level = {"run_generation.json", *(str(value) for value in object_ids)}
    if {path.name for path in output_root.iterdir()} != expected_top_level:
        raise ContractError("SAM2 output contains uncommitted top-level entries")
    outputs: dict[str, Any] = {"objects": {}}
    expected_names = {f"{index:06d}.png" for index in range(geometry.frame_count)}
    for object_id in object_ids:
        paths = object_files[object_id]
        object_dir = output_root / str(object_id)
        if object_dir.is_symlink() or not object_dir.is_dir():
            raise ContractError(f"SAM2 object {object_id} output is not a directory")
        if (
            len(paths) != geometry.frame_count
            or {path.name for path in paths} != expected_names
        ):
            raise ContractError(f"SAM2 object {object_id} frame set does not match")
        if {path.name for path in object_dir.iterdir()} != expected_names:
            raise ContractError(
                f"SAM2 object {object_id} output contains uncommitted files"
            )
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise ContractError(f"SAM2 object {object_id} masks must be regular files")
        outputs["objects"][str(object_id)] = _sam2_object_output_identity(paths)
    if manifest.get("outputs") != outputs:
        raise ContractError("SAM2 mask bytes do not match the committed generation")
    return manifest


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    status = path.stat()
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def _load_mask(path: Path, shape: tuple[int, int], *, label: str) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise ContractError(f"Could not decode {label}: {path}")
    if value.ndim == 3:
        value = value[..., 0]
    if value.shape != shape:
        raise ContractError(f"{label} shape {value.shape} != expected {shape}")
    return value > 0


def decode_v2d_inverse_depth(path: str | Path) -> np.ndarray:
    """Decode V2D's uint16 ``65535 / (depth_m + 1)`` PNG representation."""

    path = Path(path)
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise ContractError(f"Could not decode MoGe depth: {path}")
    if encoded.ndim != 2 or encoded.dtype != np.uint16:
        raise ContractError(
            f"MoGe depth must be a uint16 single-channel PNG, got "
            f"shape={encoded.shape}, dtype={encoded.dtype}"
        )
    inverse = encoded.astype(np.float32) / np.float32(65535.0)
    depth = np.full(encoded.shape, np.inf, dtype=np.float32)
    positive = encoded > 0
    depth[positive] = np.float32(1.0) / inverse[positive] - np.float32(1.0)
    return depth


def effective_e2fgvi_removal_mask(
    source_mask: np.ndarray,
    *,
    processing_width: int,
    processing_height: int,
    dilation_kernel: int,
    dilation_iterations: int,
) -> np.ndarray:
    """Reproduce E2FGVI's resize/dilate/resize mask preprocessing exactly."""

    source_mask = np.asarray(source_mask)
    if source_mask.ndim != 2 or source_mask.dtype != np.bool_:
        raise ContractError("Each source arm mask must be a 2D boolean array")
    if processing_width <= 0 or processing_height <= 0:
        raise ContractError("E2FGVI processing dimensions must be positive")
    if dilation_kernel <= 0 or dilation_kernel % 2 == 0:
        raise ContractError("E2FGVI dilation kernel must be a positive odd integer")
    if dilation_iterations < 0:
        raise ContractError("E2FGVI dilation iterations must be non-negative")
    height, width = source_mask.shape
    value = source_mask.astype(np.uint8)
    if (processing_width, processing_height) != (width, height):
        value = cv2.resize(
            value,
            (processing_width, processing_height),
            interpolation=cv2.INTER_NEAREST,
        )
    if dilation_iterations:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS, (dilation_kernel, dilation_kernel)
        )
        value = cv2.dilate(value, kernel, iterations=dilation_iterations)
    if (processing_width, processing_height) != (width, height):
        value = cv2.resize(value, (width, height), interpolation=cv2.INTER_NEAREST)
    return value.astype(bool, copy=False)


def _e2fgvi_mask_parameters(
    metadata_path: Path, geometry: VideoGeometry
) -> tuple[dict[str, Any], dict[str, int]]:
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Could not read E2FGVI metadata: {metadata_path}") from exc
    try:
        source = metadata["run"]["source_video"]
        processing = metadata["run"]["processing_resolution"]
        parameters = metadata["run"]["parameters"]
        result = {
            "processing_width": int(processing["width"]),
            "processing_height": int(processing["height"]),
            "dilation_kernel": int(parameters["dilation_kernel"]),
            "dilation_iterations": int(parameters["dilation_iterations"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(
            "E2FGVI metadata is missing mask preprocessing data"
        ) from exc
    source_geometry = VideoGeometry(
        frame_count=int(source["frame_count"]),
        width=int(source["width"]),
        height=int(source["height"]),
        fps=float(source["fps"]),
    )
    if source_geometry != geometry:
        raise ContractError(
            f"E2FGVI source geometry {source_geometry} != RGB source {geometry}"
        )
    return metadata, result


def build_rgb_estimated_occluder(
    *,
    sequence_id: str,
    source_video: str | Path,
    moge_depth_dir: str | Path,
    moge_intrinsics_dir: str | Path,
    moge_validity_dir: str | Path,
    moge_generation: str | Path,
    sam2_masks_dir: str | Path,
    sam2_object_ids: tuple[int, ...],
    sam2_prompts: str | Path,
    arm_mask: str | Path,
    e2fgvi_metadata: str | Path,
    moge_checkpoint: str | Path,
    sam2_checkpoint: str | Path,
    output_dir: str | Path,
    moge_image_id: str,
    sam2_image_id: str,
) -> dict[str, Any]:
    """Produce one immutable generic occluder bundle."""

    if not sequence_id or Path(sequence_id).name != sequence_id:
        raise ContractError("sequence_id must be one non-empty path segment")
    if not sam2_object_ids or any(
        isinstance(value, bool) or value < 0 for value in sam2_object_ids
    ):
        raise ContractError("sam2_object_ids must contain non-negative integers")
    if len(set(sam2_object_ids)) != len(sam2_object_ids):
        raise ContractError("sam2_object_ids must be unique")
    for label, value in (
        ("moge_image_id", moge_image_id),
        ("sam2_image_id", sam2_image_id),
    ):
        if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
            raise ContractError(f"{label} must be an immutable Docker image ID")

    source_video = Path(source_video).resolve()
    moge_depth_dir = Path(moge_depth_dir).resolve()
    moge_intrinsics_dir = Path(moge_intrinsics_dir).resolve()
    moge_validity_dir = Path(moge_validity_dir).resolve()
    moge_generation = Path(moge_generation).resolve()
    sam2_masks_dir = Path(sam2_masks_dir).resolve()
    sam2_prompts = Path(sam2_prompts).resolve()
    arm_mask = Path(arm_mask).resolve()
    e2fgvi_metadata = Path(e2fgvi_metadata).resolve()
    moge_checkpoint = Path(moge_checkpoint).resolve()
    sam2_checkpoint = Path(sam2_checkpoint).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Occluder output already exists: {destination}")
    for path in (
        source_video,
        sam2_prompts,
        arm_mask,
        e2fgvi_metadata,
        moge_generation,
        moge_checkpoint,
        sam2_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_record(moge_checkpoint)["sha256"] != MOGE_MODEL_SHA256:
        raise ContractError("MoGe checkpoint does not match the pinned model SHA-256")
    if file_record(sam2_checkpoint)["sha256"] != SAM2_CHECKPOINT_SHA256:
        raise ContractError("SAM2 checkpoint does not match the pinned model SHA-256")

    geometry = probe_video(source_video)
    shape = (geometry.frame_count, geometry.height, geometry.width)
    depth_files = _expected_numbered_files(
        moge_depth_dir,
        suffix=".png",
        frame_count=geometry.frame_count,
        label="MoGe depths",
    )
    intrinsics_files = _expected_numbered_files(
        moge_intrinsics_dir,
        suffix=".json",
        frame_count=geometry.frame_count,
        label="MoGe intrinsics",
    )
    validity_files = _expected_numbered_files(
        moge_validity_dir,
        suffix=".png",
        frame_count=geometry.frame_count,
        label="MoGe validity masks",
    )
    moge_run = validate_rgb_only_moge_generation(
        moge_generation,
        source_video=source_video,
        depth_files=depth_files,
        intrinsics_files=intrinsics_files,
        validity_files=validity_files,
        moge_checkpoint=moge_checkpoint,
        moge_image_id=moge_image_id,
    )
    object_files = {
        object_id: _expected_numbered_files(
            sam2_masks_dir / str(object_id),
            suffix=".png",
            frame_count=geometry.frame_count,
            label=f"SAM2 object {object_id} masks",
        )
        for object_id in sam2_object_ids
    }
    sam2_generation = sam2_masks_dir / "run_generation.json"
    if not sam2_generation.is_file():
        raise FileNotFoundError(f"Committed SAM2 generation: {sam2_generation}")
    sam2_run = validate_rgb_only_sam2_generation(
        sam2_generation,
        sequence_id=sequence_id,
        source_video=source_video,
        geometry=geometry,
        prompts_path=sam2_prompts,
        object_files=object_files,
        sam2_checkpoint=sam2_checkpoint,
        sam2_image_id=sam2_image_id,
    )
    _, mask_parameters = _e2fgvi_mask_parameters(e2fgvi_metadata, geometry)
    arm_masks = np.load(arm_mask, mmap_mode="r", allow_pickle=False)
    if arm_masks.shape != shape or arm_masks.dtype != np.bool_:
        raise ContractError(
            f"Arm mask must have shape {shape} and boolean dtype, got "
            f"shape={arm_masks.shape}, dtype={arm_masks.dtype}"
        )

    repository_root = Path(__file__).resolve().parents[1]
    implementation_paths = [repository_root / path for path in IMPLEMENTATION_FILES]
    direct_inputs = {
        "source_video": source_video,
        "sam2_prompts": sam2_prompts,
        "sam2_generation": sam2_generation,
        "arm_mask": arm_mask,
        "e2fgvi_metadata": e2fgvi_metadata,
        "moge_generation": moge_generation,
        "moge_checkpoint": moge_checkpoint,
        "sam2_checkpoint": sam2_checkpoint,
    }
    frame_inputs = [*depth_files, *intrinsics_files, *validity_files]
    for files in object_files.values():
        frame_inputs.extend(files)
    all_inputs = [*direct_inputs.values(), *frame_inputs, *implementation_paths]
    initial_signatures = {path: _stat_signature(path) for path in all_inputs}

    run_id = str(uuid.uuid4())
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{run_id}.partial"
    staging.mkdir()
    mask_path = staging / OCCLUDER_ARTIFACT_NAMES["mask"]
    depth_path = staging / OCCLUDER_ARTIFACT_NAMES["depth"]
    manifest_path = staging / "input_manifest.json"
    final_manifest_path = destination / manifest_path.name
    mask_memmap = None
    depth_memmap = None
    try:
        input_manifest = {
            "schema_version": INPUT_MANIFEST_SCHEMA,
            "sequence_id": sequence_id,
            "frame_count": geometry.frame_count,
            "moge_generation_id": moge_run["generation_id"],
            "sam2_generation_id": sam2_run["generation_id"],
            "moge_depth": _file_records(depth_files),
            "moge_intrinsics": _file_records(intrinsics_files),
            "moge_validity": _file_records(validity_files),
            "sam2_objects": {
                str(object_id): _file_records(files)
                for object_id, files in object_files.items()
            },
        }
        _write_json(manifest_path, input_manifest)
        manifest_record = file_record(
            manifest_path, recorded_path=str(final_manifest_path)
        )

        mask_memmap = np.lib.format.open_memmap(
            mask_path, mode="w+", dtype=np.bool_, shape=shape
        )
        depth_memmap = np.lib.format.open_memmap(
            depth_path, mode="w+", dtype=np.float32, shape=shape
        )
        object_pixels: list[int] = []
        removed_overlap_pixels: list[int] = []
        for frame_index in range(geometry.frame_count):
            object_mask = np.zeros(shape[1:], dtype=bool)
            for files in object_files.values():
                object_mask |= _load_mask(
                    files[frame_index], shape[1:], label="SAM2 object mask"
                )
            removal_mask = effective_e2fgvi_removal_mask(
                np.asarray(arm_masks[frame_index]), **mask_parameters
            )
            validity = _load_mask(
                validity_files[frame_index], shape[1:], label="MoGe validity mask"
            )
            depth = decode_v2d_inverse_depth(depth_files[frame_index])
            valid_depth = np.isfinite(depth) & (depth > 0.0)
            output_mask = object_mask & ~removal_mask & validity & valid_depth
            output_depth = np.full(shape[1:], np.inf, dtype=np.float32)
            output_depth[output_mask] = depth[output_mask]
            mask_memmap[frame_index] = output_mask
            depth_memmap[frame_index] = output_depth
            object_pixels.append(int(output_mask.sum()))
            removed_overlap_pixels.append(int((object_mask & removal_mask).sum()))
        mask_memmap.flush()
        depth_memmap.flush()
        del mask_memmap
        del depth_memmap
        mask_memmap = None
        depth_memmap = None

        for path, signature in initial_signatures.items():
            if _stat_signature(path) != signature:
                raise ContractError(f"Producer input changed during generation: {path}")
        metadata = {
            "schema_version": OCCLUDER_DEPTH_SCHEMA,
            "state": "complete",
            "run_id": run_id,
            "sequence_id": sequence_id,
            "completed_at": _utc_now(),
            "host_output_dir": str(destination),
            "geometry": geometry.as_dict(),
            "occluder_scope": "tool_and_target",
            "source_modalities": ["rgb"],
            "producer": {
                "name": "rgb_estimated_depth",
                "method": "MoGe-2 metric depth gated by RGB-prompted SAM2 tool/target masks",
                "version": "moge2_sam2_v1",
            },
            "depth_semantics": DEPTH_SEMANTICS,
            "artifacts": {
                "mask": f"/output/{mask_path.name}",
                "depth": f"/output/{depth_path.name}",
            },
            "artifact_bytes": {
                "mask": mask_path.stat().st_size,
                "depth": depth_path.stat().st_size,
            },
            "artifact_sha256": {
                "mask": file_record(mask_path)["sha256"],
                "depth": file_record(depth_path)["sha256"],
            },
            "provenance": {
                "schema_version": OCCLUDER_DEPTH_PROVENANCE_SCHEMA,
                "hash_algorithm": "sha256",
                "inputs": {
                    **{name: file_record(path) for name, path in direct_inputs.items()},
                    "input_manifest": manifest_record,
                },
                "implementation_sources": [
                    file_record(
                        path, recorded_path=path.relative_to(repository_root).as_posix()
                    )
                    for path in implementation_paths
                ],
            },
            "estimation": {
                "rgb_only_occluder_estimation": True,
                "uses_ground_truth": False,
                "human_initialization": "two frame-0 RGB box prompts",
                "sam2_object_ids": list(sam2_object_ids),
                "e2fgvi_removal_exclusion": mask_parameters,
                "camera_intrinsics_policy": "MoGe-estimated independently per RGB frame",
                "moge": {
                    "repository": MOGE_REPOSITORY,
                    "revision": MOGE_REVISION,
                    "source_commit": MOGE_SOURCE_COMMIT,
                    "model_sha256": MOGE_MODEL_SHA256,
                    "container_image_id": moge_image_id,
                    "generation_id": moge_run["generation_id"],
                    "generation_schema": MOGE_RUN_GENERATION_SCHEMA,
                    "input_intrinsics": None,
                },
                "sam2": {
                    "source_commit": SAM2_SOURCE_COMMIT,
                    "checkpoint_sha256": SAM2_CHECKPOINT_SHA256,
                    "container_image_id": sam2_image_id,
                    "generation_id": sam2_run["generation_id"],
                    "generation_schema": SAM2_RUN_GENERATION_SCHEMA,
                    "prompt_schema": SAM2_PROMPTS_SCHEMA,
                },
            },
            "statistics": {
                "total_occluder_pixels": int(sum(object_pixels)),
                "min_occluder_pixels_per_frame": int(min(object_pixels)),
                "max_occluder_pixels_per_frame": int(max(object_pixels)),
                "total_object_pixels_excluded_by_removal_mask": int(
                    sum(removed_overlap_pixels)
                ),
            },
        }
        _write_json(staging / OCCLUDER_METADATA_NAME, metadata)
        staging.replace(destination)
        validate_occluder_depth_bundle(
            destination / OCCLUDER_METADATA_NAME,
            geometry,
            verify_provenance_files=True,
            provenance_root=repository_root,
        )
        return metadata
    except Exception:
        if mask_memmap is not None:
            mask_memmap.flush()
        if depth_memmap is not None:
            depth_memmap.flush()
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--moge-depth-dir", required=True, type=Path)
    parser.add_argument("--moge-intrinsics-dir", required=True, type=Path)
    parser.add_argument("--moge-validity-dir", required=True, type=Path)
    parser.add_argument("--moge-generation", required=True, type=Path)
    parser.add_argument("--sam2-masks-dir", required=True, type=Path)
    parser.add_argument("--sam2-object-id", action="append", type=int, required=True)
    parser.add_argument("--sam2-prompts", required=True, type=Path)
    parser.add_argument("--arm-mask", required=True, type=Path)
    parser.add_argument("--e2fgvi-metadata", required=True, type=Path)
    parser.add_argument("--moge-checkpoint", required=True, type=Path)
    parser.add_argument("--sam2-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--moge-image-id", required=True)
    parser.add_argument("--sam2-image-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = build_rgb_estimated_occluder(
        sequence_id=args.sequence_id,
        source_video=args.source_video,
        moge_depth_dir=args.moge_depth_dir,
        moge_intrinsics_dir=args.moge_intrinsics_dir,
        moge_validity_dir=args.moge_validity_dir,
        moge_generation=args.moge_generation,
        sam2_masks_dir=args.sam2_masks_dir,
        sam2_object_ids=tuple(args.sam2_object_id),
        sam2_prompts=args.sam2_prompts,
        arm_mask=args.arm_mask,
        e2fgvi_metadata=args.e2fgvi_metadata,
        moge_checkpoint=args.moge_checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
        output_dir=args.output_dir,
        moge_image_id=args.moge_image_id,
        sam2_image_id=args.sam2_image_id,
    )
    print(
        f"Wrote RGB-estimated occluder {metadata['run_id']} -> "
        f"{metadata['host_output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
