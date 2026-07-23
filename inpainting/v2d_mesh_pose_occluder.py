"""Render Video2Data metric object tracks as a generic occluder-depth bundle.

The renderer consumes one metric mesh and one complete FoundationPose pose
sequence per task object.  FoundationPose poses map object coordinates into an
OpenCV camera frame (``+x`` right, ``+y`` down, ``+z`` forward).  Pyrender uses
an OpenGL camera frame (``+x`` right, ``+y`` up, ``-z`` forward), so object
poses are left-multiplied by ``diag(1, -1, -1, 1)`` before rasterization.

Each object is rasterized independently.  The published mask/depth is the
nearest finite union of those layers, expressed as positive metric OpenCV
camera-z with ``+inf`` outside the mask.  No ground-truth geometry, pose, or
camera calibration is consumed by this stage.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Protocol
import uuid

import numpy as np

from .contracts import (
    ContractError,
    VideoGeometry,
    validate_depth_array,
    validate_mask_array,
)
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


PRODUCER_NAME = "v2d_estimated_object"
PRODUCER_VERSION = "sam3d_foundationpose_pyrender_v1"
UPSTREAM_LINEAGE_SCHEMA = "v2d.inpainting.v2d-object-lineage/v1"
CV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])
ZNEAR_METRES = 0.001
ZFAR_METRES = 10.0
IMPLEMENTATION_FILES = (
    "inpainting/__init__.py",
    "inpainting/contracts.py",
    "inpainting/occluder_depth.py",
    "inpainting/robot_renderer/provenance.py",
    "inpainting/v2d_mesh_pose_occluder.py",
    "inpainting/video_io.py",
)

_SAFE_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MESH_SUFFIXES = frozenset((".glb", ".gltf", ".obj", ".ply", ".stl"))
_INTRINSIC_KEYS = frozenset(("fx", "fy", "cx", "cy", "width", "height"))
_FOUNDATIONPOSE_KEYS = frozenset(("rotation", "translation", "scale"))
_FILE_RECORD_KEYS = frozenset(("path", "bytes", "sha256"))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")

_LINEAGE_TOP_LEVEL_KEYS = frozenset(
    (
        "schema_version",
        "state",
        "run_id",
        "sequence_id",
        "geometry",
        "source_video",
        "rgb_evidence",
        "artifacts",
        "stages",
    )
)
_VERIFIED_SOURCE_CLAIM = {
    "uses_ground_truth": False,
    "primary_input_modalities": ["rgb"],
    "camera_intrinsics_source": (
        "legacy_stable_k_numerically_reproduced_from_moge_rgb_estimates"
    ),
    "object_mesh_source": "sam3d_estimated_from_rgb",
    "object_pose_source": "foundationpose_estimated_from_rgb_and_moge_depth",
}

MOGE_GENERATION_SCHEMA = "v2d.moge.video-to-depth-generation/v1"
MOGE_REPOSITORY = "Ruicheng/moge-2-vitl-normal"
MOGE_REVISION = "b135031bae30b5ac2ae141a0e68717795ce38340"
MOGE_SOURCE_COMMIT = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
MOGE_MODEL_BYTES = 1_323_815_904
MOGE_MODEL_SHA256 = "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01"
STABLE_INTRINSICS_GENERATION_SCHEMA = "v2d.depth.stable-intrinsics-generation/v1"
STABLE_INTRINSICS_IMPLEMENTATION_PATH = "v2d/depth/lib/stabilize_intrinsics.py"
STABLE_INTRINSICS_HISTORICAL_REPRODUCTION_IMPLEMENTATION_IDENTITY = {
    "size_bytes": 22_762,
    "sha256": "398d0a2ef36f76154ae1cacade58cf21ae6e8d9b1c32da15cfce8a174dda65bd",
}
STABLE_INTRINSICS_CURRENT_REPRODUCTION_IMPLEMENTATION_IDENTITY = {
    "size_bytes": 24_365,
    "sha256": "c566aeffaeb5feccc2c5b54961a0e4b6365b625280eca47f9145056d4d38b1b2",
}
# Compatibility names used by fixture authors and older callers that need the
# one current implementation identity rather than the accepted identity set.
STABLE_INTRINSICS_IMPLEMENTATION_BYTES = (
    STABLE_INTRINSICS_CURRENT_REPRODUCTION_IMPLEMENTATION_IDENTITY["size_bytes"]
)
STABLE_INTRINSICS_IMPLEMENTATION_SHA256 = (
    STABLE_INTRINSICS_CURRENT_REPRODUCTION_IMPLEMENTATION_IDENTITY["sha256"]
)
# The first pin validates the extant after-the-fact reproduction of the camera
# K consumed by the sequence-105 run.  The second is the exact implementation
# in this branch and is what a fresh reproduction records.  Neither identity
# says that the reproduction output itself was consumed by SAM3D/FoundationPose.
STABLE_INTRINSICS_PINNED_REPRODUCTION_IMPLEMENTATION_IDENTITIES = (
    STABLE_INTRINSICS_HISTORICAL_REPRODUCTION_IMPLEMENTATION_IDENTITY,
    STABLE_INTRINSICS_CURRENT_REPRODUCTION_IMPLEMENTATION_IDENTITY,
)
SAM2_GENERATION_SCHEMA = "v2d.sam2.video-to-masks-generation/v1"
SAM2_PROMPTS_SCHEMA = "v2d.inpainting.sam2-prompts/v1"
SAM2_CHECKPOINT_BYTES = 898_083_611
SAM2_CHECKPOINT_SHA256 = (
    "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
)

SAM3D_REPOSITORY = "facebook/sam-3d-objects"
SAM3D_REVISION = "2e73555018d2741ccd486e56c24fac41155a1dc6"
SAM3D_MOGE_REVISION = "979e84da9415762c30e6c0cf8dc0962896c793df"
FOUNDATIONPOSE_REPOSITORY = "NVlabs/FoundationPose"
FOUNDATIONPOSE_REVISION = "official_scorer_2024-01-11_refiner_2023-10-28"

# Every learned weight used by the pinned SAM3D environment.  Logical names
# disambiguate dependencies with generic basenames such as ``model.pt``.
# Values are (required basename, bytes, sha256).
SAM3D_WEIGHT_IDENTITIES = {
    "slat_decoder_gs": (
        "slat_decoder_gs.ckpt",
        171_476_155,
        "f8077c36a06eaf890dd93cda1937411f793dea1eb80b3dd9329f2038ba84a111",
    ),
    "slat_decoder_gs_4": (
        "slat_decoder_gs_4.ckpt",
        170_269_801,
        "731a0eceaa47945b52aa27f650d695b2aea9cc70945751e5609e5cb5b49f0186",
    ),
    "slat_decoder_mesh_ckpt": (
        "slat_decoder_mesh.ckpt",
        363_726_862,
        "85907b37b67d8ce5b099a96629bdcfbd873eb407dee6b3aa9a75deb15038db33",
    ),
    "slat_decoder_mesh_pt": (
        "slat_decoder_mesh.pt",
        363_728_714,
        "93333fcd57a3e36ded0b3bca6969e05ce2b35142029dadab514f41df46d2f985",
    ),
    "slat_encoder": (
        "slat_encoder.ckpt",
        173_263_986,
        "6485623145535f42c8afa4cbb68ab9953e54e2f0c1cb1eaf95dcb41051e10181",
    ),
    "slat_generator": (
        "slat_generator.ckpt",
        4_906_537_684,
        "91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb",
    ),
    "ss_decoder": (
        "ss_decoder.ckpt",
        147_609_242,
        "6dac1cd7b7fda5a38e0614fadae441f1794f80e39ea2981f1ac8aff0a7e99340",
    ),
    "ss_encoder": (
        "ss_encoder.ckpt",
        119_085_402,
        "dcc47810ac568b11fe6e4821ea1c8d6b960dfbda3e5f68e94c19f44b3bf9e83b",
    ),
    "ss_generator": (
        "ss_generator.ckpt",
        6_690_136_964,
        "225f40479e4cff4f39d6fa14c55be3abad1475bf55b61af3bec1e19ed2f6c146",
    ),
    "moge_v1": (
        "model.pt",
        1_256_823_446,
        "da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f",
    ),
    "dinov2_vitl14_reg4": (
        "dinov2_vitl14_reg4_pretrain.pth",
        1_217_607_321,
        "36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51",
    ),
    "dinov2_vitb14_reg4": (
        "dinov2_vitb14_reg4_pretrain.pth",
        346_393_545,
        "73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71",
    ),
}
FOUNDATIONPOSE_WEIGHT_IDENTITIES = {
    "scorer_config": (
        "config.yml",
        778,
        "a79db4de3b95885dd5ae86833b37b8698a75dad81e87d1086cd50b2fcd8dda3f",
    ),
    "scorer_model": (
        "model_best.pth",
        190_229_389,
        "81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26",
    ),
    "refiner_config": (
        "config.yml",
        708,
        "28a6ba94a33230ee5fc3c51939486281578b0972542bd9e38ca6123e75605686",
    ),
    "refiner_model": (
        "model_best.pth",
        68_220_109,
        "774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60",
    ),
}

_SAM3D_PARAMETER_KEYS = frozenset(
    (
        "reference_frame",
        "seed",
        "stage1_only",
        "with_mesh_postprocess",
        "with_texture_baking",
        "with_layout_postprocess",
        "use_vertex_color",
        "stage1_inference_steps",
        "geometry_grounding",
    )
)
_FP_PARAMETER_KEYS = frozenset(("scale_estimation", "tracking", "smoothing"))
_FP_SCALE_PARAMETER_KEYS = frozenset(
    (
        "lo",
        "hi",
        "n_samples",
        "n_levels",
        "iou_weight",
        "depth_weight",
        "chamfer_weight",
        "registration_iterations",
    )
)
_FP_TRACK_PARAMETER_KEYS = frozenset(
    (
        "reference_frame",
        "target_width",
        "target_height",
        "reregister_iou_thresh",
        "register_iteration",
        "track_iteration",
        "n_particles",
        "particle_process_noise_t",
        "particle_process_noise_r",
        "particle_iteration",
        "particle_mask_iou_weight",
        "mask_depth",
    )
)
_FP_SMOOTH_PARAMETER_KEYS = frozenset(
    (
        "process_noise_xy",
        "process_noise_z",
        "process_noise_r",
        "measurement_noise_xy",
        "measurement_noise_z",
        "measurement_noise_r",
        "min_iou",
    )
)


class V2DMeshPoseOccluderError(RuntimeError):
    """Raised when a Video2Data object track cannot be rendered safely."""


@dataclass(frozen=True)
class CameraModel:
    """One pinhole camera shared by every object and frame."""

    path: Path
    matrix: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class MetricObjectTrack:
    """One metric mesh and its complete object-to-camera SE(3) trajectory."""

    name: str
    mesh_path: Path
    poses_dir: Path
    pose_paths: tuple[Path, ...]
    object_to_camera: np.ndarray
    pose_format: str


@dataclass(frozen=True)
class UpstreamObjectLineage:
    """Validated immutable claim binding RGB-only V2D stages to their outputs."""

    path: Path
    manifest: dict[str, Any]
    record: dict[str, Any]
    verified_source_claim: dict[str, Any]


@dataclass(frozen=True)
class V2DMeshPoseInputs:
    """Validated, immutable inputs for one render."""

    sequence_id: str
    source_video: Path
    geometry: VideoGeometry
    camera: CameraModel
    objects: tuple[MetricObjectTrack, ...]
    upstream_lineage: UpstreamObjectLineage
    provenance_inputs: dict[str, dict[str, Any]]
    input_signatures: dict[Path, tuple[int, int, int, int]]


class DepthLayerRenderer(Protocol):
    """Small injectable boundary around the GPU-dependent rasterizer."""

    def render(self, object_index: int, object_to_camera: np.ndarray) -> np.ndarray:
        """Return one ``(H,W)`` pyrender depth layer (zero is background)."""

    def close(self) -> None:
        """Release renderer resources."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    status = path.stat()
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Could not read {label}: {path}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        _write_json(temporary, payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _numeric(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ContractError(f"{label} must be finite")
    return result


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{label} must be a non-negative integer")
    return value


def _exact_object(value: object, keys: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContractError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _lineage_file_record(value: object, *, label: str) -> dict[str, Any]:
    record = _exact_object(value, _FILE_RECORD_KEYS, label=label)
    recorded_path = record["path"]
    if not isinstance(recorded_path, str) or not recorded_path:
        raise ContractError(f"{label} path must be a non-empty string")
    if not Path(recorded_path).is_absolute():
        raise ContractError(f"{label} path must be absolute")
    byte_count = record["bytes"]
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
    ):
        raise ContractError(f"{label} bytes must be a positive integer")
    digest = record["sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ContractError(f"{label} sha256 must be 64 lowercase hexadecimal digits")
    return record


def _match_lineage_file(
    value: object, actual_path: Path, *, label: str
) -> dict[str, Any]:
    record = _lineage_file_record(value, label=label)
    if Path(record["path"]).name != actual_path.name:
        raise ContractError(
            f"{label} basename {Path(record['path']).name!r} does not match "
            f"{actual_path.name!r}"
        )
    actual = file_record(actual_path)
    if record["bytes"] != actual["bytes"] or record["sha256"] != actual["sha256"]:
        raise ContractError(f"{label} does not fingerprint the consumed file")
    return record


def _validate_container_identity(value: object, *, label: str) -> dict[str, Any]:
    container = _exact_object(
        value, frozenset(("image", "image_id")), label=f"{label} container"
    )
    if not isinstance(container["image"], str) or not container["image"].strip():
        raise ContractError(f"{label} container image must be non-empty")
    image_id = container["image_id"]
    if not isinstance(image_id, str) or _IMAGE_ID.fullmatch(image_id) is None:
        raise ContractError(f"{label} container image_id must be immutable sha256")
    return container


def _validate_weight_identities(
    value: object,
    expected: dict[str, tuple[str, int, str]],
    *,
    label: str,
) -> None:
    if not isinstance(value, list) or len(value) != len(expected):
        raise ContractError(
            f"{label} weights must list exactly {len(expected)} pinned artifacts"
        )
    names: list[str] = []
    for index, item in enumerate(value):
        item = _exact_object(
            item,
            frozenset(("name", "record")),
            label=f"{label} weight {index}",
        )
        name = item["name"]
        if not isinstance(name, str) or name not in expected:
            raise ContractError(f"{label} weight {index} has unknown logical name")
        names.append(name)
        record = _lineage_file_record(item["record"], label=f"{label} weight {name}")
        basename, byte_count, digest = expected[name]
        if (
            Path(record["path"]).name != basename
            or record["bytes"] != byte_count
            or record["sha256"] != digest
        ):
            raise ContractError(
                f"{label} weight {name} identity does not match the pin"
            )
    if names != sorted(expected):
        raise ContractError(
            f"{label} weights must use deterministic sorted logical order"
        )


def _finite_parameter(value: object, *, label: str, minimum: float = 0.0) -> float:
    result = _numeric(value, label=label)
    if result < minimum:
        raise ContractError(f"{label} must be at least {minimum}")
    return result


def _validate_sam3d_parameters(
    value: object, *, object_names: tuple[str, ...], geometry: VideoGeometry
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(object_names):
        raise ContractError(
            "SAM3D parameters must contain exactly one entry per object"
        )
    for name in object_names:
        parameters = _exact_object(
            value[name], _SAM3D_PARAMETER_KEYS, label=f"SAM3D parameters for {name}"
        )
        reference_frame = _nonnegative_integer(
            parameters["reference_frame"],
            label=f"SAM3D {name} reference_frame",
        )
        if reference_frame >= geometry.frame_count:
            raise ContractError(f"SAM3D {name} reference_frame is outside the video")
        _nonnegative_integer(parameters["seed"], label=f"SAM3D {name} seed")
        for key in (
            "stage1_only",
            "with_mesh_postprocess",
            "with_texture_baking",
            "with_layout_postprocess",
            "use_vertex_color",
        ):
            if type(parameters[key]) is not bool:
                raise ContractError(f"SAM3D {name} {key} must be boolean")
        if parameters["stage1_only"]:
            raise ContractError("SAM3D stage1_only cannot produce the required mesh")
        inference_steps = parameters["stage1_inference_steps"]
        if inference_steps is not None:
            _positive_integer(
                inference_steps, label=f"SAM3D {name} stage1_inference_steps"
            )
        if parameters["geometry_grounding"] != "moge_depth":
            raise ContractError(
                f"SAM3D {name} geometry_grounding must be the bound MoGe RGB depth"
            )
    return value


def _validate_foundationpose_parameters(
    value: object,
    *,
    object_names: tuple[str, ...],
    geometry: VideoGeometry,
    sam3d_parameters: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(object_names):
        raise ContractError(
            "FoundationPose parameters must contain exactly one entry per object"
        )
    for name in object_names:
        parameters = _exact_object(
            value[name],
            _FP_PARAMETER_KEYS,
            label=f"FoundationPose parameters for {name}",
        )
        scale = _exact_object(
            parameters["scale_estimation"],
            _FP_SCALE_PARAMETER_KEYS,
            label=f"FoundationPose scale parameters for {name}",
        )
        lo = _finite_parameter(scale["lo"], label=f"FoundationPose {name} scale lo")
        hi = _finite_parameter(scale["hi"], label=f"FoundationPose {name} scale hi")
        if lo <= 0.0 or hi <= lo:
            raise ContractError(f"FoundationPose {name} scale bounds are invalid")
        _positive_integer(
            scale["n_samples"], label=f"FoundationPose {name} scale n_samples"
        )
        _positive_integer(
            scale["n_levels"], label=f"FoundationPose {name} scale n_levels"
        )
        for key in ("iou_weight", "depth_weight", "chamfer_weight"):
            _finite_parameter(scale[key], label=f"FoundationPose {name} scale {key}")
        _positive_integer(
            scale["registration_iterations"],
            label=f"FoundationPose {name} scale registration_iterations",
        )

        tracking = _exact_object(
            parameters["tracking"],
            _FP_TRACK_PARAMETER_KEYS,
            label=f"FoundationPose tracking parameters for {name}",
        )
        reference_frame = _nonnegative_integer(
            tracking["reference_frame"],
            label=f"FoundationPose {name} reference_frame",
        )
        if reference_frame >= geometry.frame_count:
            raise ContractError(
                f"FoundationPose {name} reference_frame is outside the video"
            )
        if reference_frame != sam3d_parameters[name]["reference_frame"]:
            raise ContractError(
                f"FoundationPose and SAM3D reference frames differ for {name}"
            )
        target_width = tracking["target_width"]
        target_height = tracking["target_height"]
        if (target_width is None) != (target_height is None):
            raise ContractError(
                f"FoundationPose {name} target width/height must both be null or set"
            )
        if target_width is not None:
            if (
                _positive_integer(
                    target_width, label=f"FoundationPose {name} target_width"
                )
                != geometry.width
                or _positive_integer(
                    target_height, label=f"FoundationPose {name} target_height"
                )
                != geometry.height
            ):
                raise ContractError(
                    f"FoundationPose {name} target resolution must match the video"
                )
        threshold = tracking["reregister_iou_thresh"]
        if threshold is not None:
            threshold = _finite_parameter(
                threshold,
                label=f"FoundationPose {name} reregister_iou_thresh",
            )
            if threshold > 1.0:
                raise ContractError(
                    f"FoundationPose {name} reregister_iou_thresh must be <= 1"
                )
        for key in (
            "register_iteration",
            "track_iteration",
            "n_particles",
            "particle_iteration",
        ):
            _positive_integer(tracking[key], label=f"FoundationPose {name} {key}")
        for key in (
            "particle_process_noise_t",
            "particle_process_noise_r",
            "particle_mask_iou_weight",
        ):
            _finite_parameter(tracking[key], label=f"FoundationPose {name} {key}")
        if type(tracking["mask_depth"]) is not bool:
            raise ContractError(f"FoundationPose {name} mask_depth must be boolean")

        smoothing = _exact_object(
            parameters["smoothing"],
            _FP_SMOOTH_PARAMETER_KEYS,
            label=f"FoundationPose smoothing parameters for {name}",
        )
        for key in _FP_SMOOTH_PARAMETER_KEYS - {"min_iou"}:
            _finite_parameter(
                smoothing[key], label=f"FoundationPose {name} smoothing {key}"
            )
        min_iou = _finite_parameter(
            smoothing["min_iou"],
            label=f"FoundationPose {name} smoothing min_iou",
        )
        if not 0.0 < min_iou <= 1.0:
            raise ContractError(
                f"FoundationPose {name} smoothing min_iou must be in (0, 1]"
            )
    return value


def _validate_stage_identity(
    value: object,
    *,
    label: str,
    repository: str,
    revision: str,
    weight_identities: dict[str, tuple[str, int, str]],
) -> dict[str, Any]:
    stage = _exact_object(
        value,
        frozenset(("model", "container", "weights", "parameters")),
        label=f"{label} stage",
    )
    model = _exact_object(
        stage["model"],
        frozenset(("repository", "revision")),
        label=f"{label} model",
    )
    if model != {"repository": repository, "revision": revision}:
        raise ContractError(f"{label} model identity does not match the pinned model")
    _validate_container_identity(stage["container"], label=label)
    _validate_weight_identities(stage["weights"], weight_identities, label=label)
    return stage


def _embedded_generation_record(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not {"sha256", "size_bytes"} <= set(value):
        raise ContractError(f"{label} must contain a size and SHA-256")
    if set(value) not in (
        {"sha256", "size_bytes"},
        {"path", "sha256", "size_bytes"},
    ):
        raise ContractError(f"{label} may contain only path, size_bytes, and sha256")
    record = value
    if (
        not isinstance(record["sha256"], str)
        or _SHA256.fullmatch(record["sha256"]) is None
        or isinstance(record["size_bytes"], bool)
        or not isinstance(record["size_bytes"], int)
        or record["size_bytes"] <= 0
    ):
        raise ContractError(f"{label} has an invalid size/hash record")
    if "path" in record and (
        not isinstance(record["path"], str)
        or not record["path"]
        or not Path(record["path"]).is_absolute()
    ):
        raise ContractError(f"{label} path must be a non-empty absolute path")
    return record


def _match_embedded_generation_record(value: object, path: Path, *, label: str) -> None:
    record = _embedded_generation_record(value, label=label)
    if "path" in record and Path(record["path"]).name != path.name:
        raise ContractError(f"{label} names a different artifact")
    actual = file_record(path)
    if record["size_bytes"] != actual["bytes"] or record["sha256"] != actual["sha256"]:
        raise ContractError(f"{label} does not match the live generation artifact")


def _match_sam2_prompt_identity(value: object, prompts: dict[str, Any]) -> None:
    """Match SAM2's canonicalized prompt bytes, which omit the source newline."""

    record = _embedded_generation_record(value, label="SAM2 prompts identity")
    canonical = json.dumps(prompts, indent=2).encode("utf-8")
    if (
        record["size_bytes"] != len(canonical)
        or record["sha256"] != hashlib.sha256(canonical).hexdigest()
    ):
        raise ContractError(
            "SAM2 prompts identity does not match wrapper-canonicalized RGB prompts"
        )


def _resolve_lineage_live_file(
    value: object,
    *,
    anchors: Sequence[tuple[Path, Path]],
    label: str,
) -> Path:
    """Resolve a recorded host path through known container/host path anchors."""

    record = _lineage_file_record(value, label=label)
    recorded = Path(record["path"])
    candidates = [recorded]
    for recorded_anchor, live_anchor in anchors:
        for recorded_base, live_base in zip(
            recorded_anchor.parents, live_anchor.parents, strict=False
        ):
            try:
                relative = recorded.relative_to(recorded_base)
            except ValueError:
                continue
            candidates.append(live_base / relative)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        try:
            _match_lineage_file(record, candidate, label=label)
        except ContractError:
            continue
        return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve {label}: {recorded}")


def _verify_moge_rgb_evidence(
    generation_path: Path,
    intrinsics_records: object,
    *,
    source_video: Path,
    geometry: VideoGeometry,
    camera: CameraModel,
    anchors: Sequence[tuple[Path, Path]],
) -> None:
    generation = _read_json(generation_path, label="MoGe generation manifest")
    if not isinstance(generation, dict):
        raise ContractError("MoGe generation manifest must be an object")
    if (
        generation.get("schema_version") != MOGE_GENERATION_SCHEMA
        or generation.get("state") != "complete"
    ):
        raise ContractError("MoGe generation must be a complete pinned v1 run")
    parameters = generation.get("parameters")
    required_parameters = {
        "input_intrinsics_path": None,
        "intrinsics_mode": "estimated_from_rgb",
        "frame_index_origin": 0,
    }
    if not isinstance(parameters, dict) or any(
        parameters.get(key) != expected for key, expected in required_parameters.items()
    ):
        raise ContractError(
            "MoGe generation must estimate intrinsics from RGB with no input calibration"
        )
    if set(parameters.get("requested_outputs", ())) != {
        "depth",
        "intrinsics",
        "mask",
        "points",
    }:
        raise ContractError("MoGe generation does not declare the required outputs")
    sources = generation.get("sources")
    if not isinstance(sources, dict) or sources.get("input_intrinsics") is not None:
        raise ContractError("MoGe generation contains an external intrinsic source")
    _match_embedded_generation_record(
        sources.get("video"), source_video, label="MoGe RGB source"
    )
    revisions = generation.get("source_revisions")
    if revisions != {
        "moge_repository": MOGE_REPOSITORY,
        "moge_huggingface_revision": MOGE_REVISION,
        "moge_source_commit": MOGE_SOURCE_COMMIT,
    }:
        raise ContractError("MoGe generation source identity is not pinned")
    model = generation.get("model")
    checkpoint = model.get("checkpoint") if isinstance(model, dict) else None
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("size_bytes") != MOGE_MODEL_BYTES
        or checkpoint.get("sha256") != MOGE_MODEL_SHA256
    ):
        raise ContractError("MoGe generation checkpoint identity is not pinned")
    image_id = generation.get("execution_environment", {}).get("container_image_id")
    if not isinstance(image_id, str) or _IMAGE_ID.fullmatch(image_id) is None:
        raise ContractError("MoGe generation container image is not immutable")
    if generation.get("expected_frames") != {
        "count": geometry.frame_count,
        "indices": [0, geometry.frame_count - 1],
    }:
        raise ContractError("MoGe generation frame domain does not match the video")

    if (
        not isinstance(intrinsics_records, list)
        or len(intrinsics_records) != geometry.frame_count
    ):
        raise ContractError("RGB evidence must bind every MoGe intrinsics estimate")
    output = generation.get("outputs", {}).get("intrinsics")
    files = output.get("files") if isinstance(output, dict) else None
    if not isinstance(files, dict) or len(files) != geometry.frame_count:
        raise ContractError(
            "MoGe generation does not fingerprint every intrinsic output"
        )
    matrices: list[np.ndarray] = []
    for frame_index, record in enumerate(intrinsics_records):
        expected_name = f"{frame_index:06d}.json"
        record = _lineage_file_record(
            record, label=f"MoGe intrinsics {frame_index:06d}"
        )
        path = _resolve_lineage_live_file(
            record,
            anchors=anchors,
            label=f"MoGe intrinsics {frame_index:06d}",
        )
        if path.name != expected_name:
            raise ContractError("MoGe intrinsics evidence is not in frame order")
        _match_lineage_file(record, path, label=f"MoGe intrinsics {frame_index:06d}")
        embedded = files.get(expected_name)
        _match_embedded_generation_record(
            embedded, path, label=f"MoGe generation intrinsic {frame_index:06d}"
        )
        model = load_camera_model(
            path, expected_width=geometry.width, expected_height=geometry.height
        )
        matrices.append(model.matrix)
    values = np.stack(matrices)
    stable = np.array(
        [
            [np.median(values[:, 0, 0]), 0.0, np.median(values[:, 0, 2])],
            [0.0, np.median(values[:, 1, 1]), np.median(values[:, 1, 2])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not np.allclose(stable, camera.matrix, atol=1e-9, rtol=0.0):
        raise ContractError(
            "Stable camera K is not the temporal median of the bound MoGe RGB estimates"
        )


def _pinned_stable_intrinsics_implementation_identity(
    value: object,
) -> dict[str, object]:
    expected_keys = {"path", "size_bytes", "sha256"}
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("path") != STABLE_INTRINSICS_IMPLEMENTATION_PATH
    ):
        raise ContractError(
            "Stable-intrinsics reproduction implementation is not pinned"
        )
    identity = {
        "size_bytes": value["size_bytes"],
        "sha256": value["sha256"],
    }
    if (
        identity
        not in STABLE_INTRINSICS_PINNED_REPRODUCTION_IMPLEMENTATION_IDENTITIES
    ):
        raise ContractError(
            "Stable-intrinsics reproduction implementation is not pinned"
        )
    return identity


def _verify_stable_intrinsics_reproduction(
    reproduction_path: Path,
    *,
    moge_generation_path: Path,
    intrinsics_records: object,
    camera: CameraModel,
    geometry: VideoGeometry,
    anchors: Sequence[tuple[Path, Path]],
) -> None:
    reproduction = _read_json(
        reproduction_path, label="stable-intrinsics reproduction manifest"
    )
    if not isinstance(reproduction, dict) or (
        reproduction.get("schema_version") != STABLE_INTRINSICS_GENERATION_SCHEMA
        or reproduction.get("state") != "complete"
    ):
        raise ContractError(
            "Stable-intrinsics reproduction must be a complete pinned v1 run"
        )
    expected_parameters = {
        "algorithm": "coordinate_wise_temporal_median/v1",
        "dimension_policy": "require_constant_across_frames",
        "fix_principal_point": True,
        "frame_order": "contiguous_zero_based_six_digit_filenames",
        "principal_point_policy": "image_center",
    }
    if reproduction.get("parameters") != expected_parameters:
        raise ContractError("Stable-intrinsics reproduction parameters are not pinned")
    implementation = reproduction.get("implementation_sources", {}).get(
        STABLE_INTRINSICS_IMPLEMENTATION_PATH
    )
    implementation_identity = _pinned_stable_intrinsics_implementation_identity(
        implementation
    )

    moge_generation = _read_json(
        moge_generation_path, label="bound MoGe generation manifest"
    )
    sources = reproduction.get("sources")
    if not isinstance(sources, dict):
        raise ContractError("Stable-intrinsics reproduction is missing sources")
    if sources.get("moge_schema_version") != MOGE_GENERATION_SCHEMA or sources.get(
        "moge_generation_id"
    ) != moge_generation.get("generation_id"):
        raise ContractError("Stable-intrinsics reproduction names a different MoGe run")
    _match_embedded_generation_record(
        sources.get("moge_generation_manifest"),
        moge_generation_path,
        label="Stable-intrinsics source MoGe manifest",
    )
    source_intrinsics = sources.get("intrinsics")
    source_files = (
        source_intrinsics.get("files") if isinstance(source_intrinsics, dict) else None
    )
    if (
        not isinstance(intrinsics_records, list)
        or not isinstance(source_files, dict)
        or len(source_files) != geometry.frame_count
    ):
        raise ContractError(
            "Stable-intrinsics reproduction does not bind every MoGe estimate"
        )
    for frame_index, record in enumerate(intrinsics_records):
        path = _resolve_lineage_live_file(
            record,
            anchors=anchors,
            label=f"Stable-intrinsics source {frame_index:06d}",
        )
        _match_embedded_generation_record(
            source_files.get(f"{frame_index:06d}.json"),
            path,
            label=f"Stable-intrinsics source {frame_index:06d}",
        )

    static_identity = reproduction.get("static_identity")
    if not isinstance(static_identity, dict):
        raise ContractError("Stable-intrinsics reproduction lacks static identity")
    if (
        static_identity.get("moge_generation_id") != sources.get("moge_generation_id")
        or static_identity.get("moge_schema_version") != MOGE_GENERATION_SCHEMA
        or static_identity.get("parameters") != expected_parameters
        or static_identity.get("intrinsics", {}).get("files") != source_files
        or static_identity.get("implementation_sources", {}).get(
            STABLE_INTRINSICS_IMPLEMENTATION_PATH
        )
        != implementation_identity
    ):
        raise ContractError("Stable-intrinsics static identity is inconsistent")

    output = reproduction.get("output")
    stable_record = (
        output.get("stable_intrinsics") if isinstance(output, dict) else None
    )
    embedded = _embedded_generation_record(
        stable_record, label="Reproduced stable intrinsics"
    )
    lineage_record = {
        "path": embedded["path"],
        "bytes": embedded["size_bytes"],
        "sha256": embedded["sha256"],
    }
    reproduced_path = _resolve_lineage_live_file(
        lineage_record,
        anchors=[*anchors, (reproduction_path, reproduction_path)],
        label="Reproduced stable intrinsics",
    )
    reproduced = load_camera_model(
        reproduced_path,
        expected_width=geometry.width,
        expected_height=geometry.height,
    )
    if not np.array_equal(reproduced.matrix, camera.matrix):
        raise ContractError(
            "RGB-only stable-intrinsics reproduction differs from the legacy K "
            "actually consumed by SAM3D/FoundationPose"
        )
    expected_values = {
        "fx": float(camera.matrix[0, 0]),
        "fy": float(camera.matrix[1, 1]),
        "cx": float(camera.matrix[0, 2]),
        "cy": float(camera.matrix[1, 2]),
        "width": camera.width,
        "height": camera.height,
    }
    if output.get("values") != expected_values:
        raise ContractError("Stable-intrinsics reproduction values are inconsistent")


def _verify_sam2_rgb_evidence(
    generation_path: Path,
    prompts_path: Path,
    *,
    source_video: Path,
    geometry: VideoGeometry,
    sequence_id: str,
) -> set[int]:
    prompts = _read_json(prompts_path, label="SAM2 RGB prompts")
    if not isinstance(prompts, dict):
        raise ContractError("SAM2 prompts must be an object")
    metadata = prompts.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != SAM2_PROMPTS_SCHEMA
        or metadata.get("sequence_id") != sequence_id
        or metadata.get("geometry") != geometry.as_dict()
        or metadata.get("role") != "rgb_only_tool_and_target_segmentation"
        or metadata.get("initialization") != "human_box_prompts_on_rgb_frame_0"
    ):
        raise ContractError("SAM2 prompts are not the audited RGB-only object prompts")
    if Path(str(metadata.get("source_video"))).resolve() != source_video.resolve():
        raise ContractError("SAM2 prompts reference a different source video")
    prompt_values = prompts.get("prompts")
    if not isinstance(prompt_values, list) or not prompt_values:
        raise ContractError("SAM2 prompts contain no objects")
    object_ids: set[int] = set()
    for prompt in prompt_values:
        if (
            not isinstance(prompt, dict)
            or prompt.get("frame_index") != 0
            or prompt.get("mask_path") is not None
            or prompt.get("box") is None
            or prompt.get("points") is not None
            or prompt.get("point_labels") is not None
            or isinstance(prompt.get("object_id"), bool)
            or not isinstance(prompt.get("object_id"), int)
        ):
            raise ContractError("SAM2 object initialization must use frame-0 RGB boxes")
        box = prompt["box"]
        if not isinstance(box, dict) or set(box) != {"x0", "y0", "x1", "y1"}:
            raise ContractError("SAM2 RGB box must contain x0, y0, x1, and y1")
        x0 = _numeric(box["x0"], label="SAM2 box x0")
        y0 = _numeric(box["y0"], label="SAM2 box y0")
        x1 = _numeric(box["x1"], label="SAM2 box x1")
        y1 = _numeric(box["y1"], label="SAM2 box y1")
        if not (0.0 <= x0 < x1 <= geometry.width and 0.0 <= y0 < y1 <= geometry.height):
            raise ContractError("SAM2 RGB box lies outside the source frame")
        object_ids.add(prompt["object_id"])
    prompt_labels = metadata.get("object_ids")
    if (
        not isinstance(prompt_labels, dict)
        or set(prompt_labels) != {str(item) for item in object_ids}
        or not all(isinstance(value, str) and value for value in prompt_labels.values())
    ):
        raise ContractError("SAM2 prompt object labels do not match the RGB boxes")

    generation = _read_json(generation_path, label="SAM2 generation manifest")
    if not isinstance(generation, dict) or (
        generation.get("schema_version") != SAM2_GENERATION_SCHEMA
        or generation.get("state") != "complete"
    ):
        raise ContractError("SAM2 generation must be a complete pinned v1 run")
    if generation.get("expected") != {
        "frame_count": geometry.frame_count,
        "object_ids": sorted(object_ids),
    }:
        raise ContractError(
            "SAM2 generation frame/object domain does not match prompts"
        )
    identity = generation.get("static_identity")
    if not isinstance(identity, dict):
        raise ContractError("SAM2 generation is missing static identity")
    _match_sam2_prompt_identity(identity.get("prompts_json"), prompts)
    video_identity = identity.get("video")
    if not isinstance(video_identity, dict) or video_identity.get("kind") != "file":
        raise ContractError("SAM2 generation does not identify one RGB video")
    _match_embedded_generation_record(
        video_identity.get("artifact"), source_video, label="SAM2 RGB source"
    )
    checkpoint = identity.get("checkpoint")
    artifact = checkpoint.get("artifact") if isinstance(checkpoint, dict) else None
    if (
        not isinstance(artifact, dict)
        or artifact.get("size_bytes") != SAM2_CHECKPOINT_BYTES
        or artifact.get("sha256") != SAM2_CHECKPOINT_SHA256
    ):
        raise ContractError("SAM2 checkpoint identity is not pinned")
    image_id = identity.get("execution_environment", {}).get("container_image_id")
    if not isinstance(image_id, str) or _IMAGE_ID.fullmatch(image_id) is None:
        raise ContractError("SAM2 generation container image is not immutable")

    outputs = generation.get("outputs", {}).get("objects")
    if not isinstance(outputs, dict) or set(outputs) != {
        str(item) for item in object_ids
    }:
        raise ContractError("SAM2 generation outputs do not match prompted objects")
    for object_id in object_ids:
        output = outputs[str(object_id)]
        files = output.get("files") if isinstance(output, dict) else None
        if not isinstance(files, dict) or len(files) != geometry.frame_count:
            raise ContractError(f"SAM2 object {object_id} output is incomplete")
        for frame_index in range(geometry.frame_count):
            filename = f"{frame_index:06d}.png"
            path = generation_path.parent / str(object_id) / filename
            _match_embedded_generation_record(
                files.get(filename),
                path,
                label=f"SAM2 object {object_id} mask {frame_index:06d}",
            )
    return object_ids


def _validate_sam3d_transform_artifact(path: Path, *, object_name: str) -> None:
    value = _exact_object(
        _read_json(path, label=f"{object_name} SAM3D transform"),
        _FOUNDATIONPOSE_KEYS,
        label=f"{object_name} SAM3D transform",
    )
    try:
        quaternion = np.asarray(value["rotation"], dtype=np.float64)
        translation = np.asarray(value["translation"], dtype=np.float64)
        scale = np.asarray(value["scale"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{object_name} SAM3D transform must be numeric") from exc
    if quaternion.shape != (4,) or translation.shape != (3,) or scale.shape != (3,):
        raise ContractError(f"{object_name} SAM3D transform has invalid field shapes")
    if not (
        np.isfinite(quaternion).all()
        and np.isfinite(translation).all()
        and np.isfinite(scale).all()
    ):
        raise ContractError(f"{object_name} SAM3D transform must be finite")
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-3, rtol=0.0):
        raise ContractError(f"{object_name} SAM3D quaternion is not unit length")
    if np.any(scale <= 0.0) or not np.allclose(scale, scale[0], atol=1e-6, rtol=0.0):
        raise ContractError(f"{object_name} SAM3D scale must be positive and isotropic")


def _validate_metric_scale_artifact(path: Path, *, object_name: str) -> None:
    value = _exact_object(
        _read_json(path, label=f"{object_name} metric scale"),
        frozenset(("scale",)),
        label=f"{object_name} metric scale",
    )
    if _numeric(value["scale"], label=f"{object_name} metric scale") <= 0.0:
        raise ContractError(f"{object_name} metric scale must be positive")


def load_upstream_object_lineage(
    manifest_path: str | Path,
    *,
    sequence_id: str,
    source_video: Path,
    geometry: VideoGeometry,
    camera: CameraModel,
    objects: tuple[MetricObjectTrack, ...],
) -> UpstreamObjectLineage:
    """Validate a complete RGB-only V2D lineage attestation against live inputs."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Upstream V2D lineage manifest: {path}")
    manifest = _read_json(path, label="upstream V2D lineage manifest")
    manifest = _exact_object(
        manifest, _LINEAGE_TOP_LEVEL_KEYS, label="Upstream V2D lineage manifest"
    )
    if manifest["schema_version"] != UPSTREAM_LINEAGE_SCHEMA:
        raise ContractError(
            f"Upstream lineage schema must be {UPSTREAM_LINEAGE_SCHEMA!r}"
        )
    if manifest["state"] != "complete":
        raise ContractError("Upstream lineage state must be 'complete'")
    try:
        run_id = str(uuid.UUID(str(manifest["run_id"])))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ContractError("Upstream lineage run_id must be a canonical UUID") from exc
    if run_id != manifest["run_id"]:
        raise ContractError("Upstream lineage run_id must be a canonical UUID")
    if manifest["sequence_id"] != sequence_id:
        raise ContractError("Upstream lineage sequence_id does not match the render")
    if manifest["geometry"] != geometry.as_dict():
        raise ContractError("Upstream lineage geometry does not match the source video")
    source_record = _match_lineage_file(
        manifest["source_video"], source_video, label="Upstream source video"
    )

    artifacts = _exact_object(
        manifest["artifacts"],
        frozenset(("intrinsics", "objects")),
        label="Upstream lineage artifacts",
    )
    intrinsic_record = _match_lineage_file(
        artifacts["intrinsics"], camera.path, label="Upstream estimated intrinsics"
    )
    anchors: list[tuple[Path, Path]] = [
        (Path(source_record["path"]), source_video),
        (Path(intrinsic_record["path"]), camera.path),
    ]
    object_names = tuple(track.name for track in objects)
    object_values = artifacts["objects"]
    if not isinstance(object_values, list) or len(object_values) != len(objects):
        raise ContractError("Upstream lineage must bind exactly every rendered object")
    object_lineage: list[tuple[MetricObjectTrack, dict[str, Any], int]] = []
    for track, value in zip(objects, object_values, strict=True):
        value = _exact_object(
            value,
            frozenset(("name", "sam2_object_id", "mesh", "poses", "chain")),
            label=f"Upstream lineage object {track.name}",
        )
        if value["name"] != track.name:
            raise ContractError(
                "Upstream lineage objects must use sorted renderer order"
            )
        object_id = _positive_integer(
            value["sam2_object_id"], label=f"Upstream {track.name} SAM2 object_id"
        )
        mesh_record = _match_lineage_file(
            value["mesh"], track.mesh_path, label=f"Upstream {track.name} metric mesh"
        )
        anchors.append((Path(mesh_record["path"]), track.mesh_path))
        poses = value["poses"]
        if not isinstance(poses, list) or len(poses) != len(track.pose_paths):
            raise ContractError(
                f"Upstream lineage must bind every {track.name} pose file"
            )
        for frame_index, (record, pose_path) in enumerate(
            zip(poses, track.pose_paths, strict=True)
        ):
            pose_record = _match_lineage_file(
                record,
                pose_path,
                label=f"Upstream {track.name} pose {frame_index:06d}",
            )
            anchors.append((Path(pose_record["path"]), pose_path))
        object_lineage.append((track, value, object_id))

    rgb_evidence = _exact_object(
        manifest["rgb_evidence"],
        frozenset(
            (
                "moge_generation",
                "moge_intrinsics",
                "stable_intrinsics_reproduction",
                "sam2_generation",
                "sam2_prompts",
            )
        ),
        label="Upstream RGB evidence",
    )
    moge_generation = _resolve_lineage_live_file(
        rgb_evidence["moge_generation"],
        anchors=anchors,
        label="MoGe generation manifest",
    )
    sam2_generation = _resolve_lineage_live_file(
        rgb_evidence["sam2_generation"],
        anchors=anchors,
        label="SAM2 generation manifest",
    )
    sam2_prompts = _resolve_lineage_live_file(
        rgb_evidence["sam2_prompts"],
        anchors=anchors,
        label="SAM2 RGB prompts",
    )
    stable_intrinsics_reproduction = _resolve_lineage_live_file(
        rgb_evidence["stable_intrinsics_reproduction"],
        anchors=anchors,
        label="Stable-intrinsics reproduction manifest",
    )
    evidence_anchors = [
        *anchors,
        (Path(rgb_evidence["moge_generation"]["path"]), moge_generation),
        (Path(rgb_evidence["sam2_generation"]["path"]), sam2_generation),
        (Path(rgb_evidence["sam2_prompts"]["path"]), sam2_prompts),
        (
            Path(rgb_evidence["stable_intrinsics_reproduction"]["path"]),
            stable_intrinsics_reproduction,
        ),
    ]
    _verify_moge_rgb_evidence(
        moge_generation,
        rgb_evidence["moge_intrinsics"],
        source_video=source_video,
        geometry=geometry,
        camera=camera,
        anchors=evidence_anchors,
    )
    _verify_stable_intrinsics_reproduction(
        stable_intrinsics_reproduction,
        moge_generation_path=moge_generation,
        intrinsics_records=rgb_evidence["moge_intrinsics"],
        camera=camera,
        geometry=geometry,
        anchors=evidence_anchors,
    )
    prompted_object_ids = _verify_sam2_rgb_evidence(
        sam2_generation,
        sam2_prompts,
        source_video=source_video,
        geometry=geometry,
        sequence_id=sequence_id,
    )
    lineage_object_ids = {item[2] for item in object_lineage}
    if len(lineage_object_ids) != len(object_lineage):
        raise ContractError("Each rendered object must bind a distinct SAM2 object_id")
    if lineage_object_ids != prompted_object_ids:
        raise ContractError(
            "Rendered objects do not exactly match the SAM2 RGB prompts"
        )

    chain_keys = frozenset(
        (
            "sam3d_mesh",
            "sam3d_transform",
            "mesh_pretransformed",
            "scale",
            "scale_registration_pose",
            "raw_poses",
        )
    )
    for track, value, _object_id in object_lineage:
        root = track.mesh_path.parent
        chain = _exact_object(
            value["chain"], chain_keys, label=f"Upstream {track.name} artifact chain"
        )
        fixed_paths = {
            "sam3d_mesh": root / "sam3d_mesh.glb",
            "sam3d_transform": root / "sam3d_transform.json",
            "mesh_pretransformed": root / "mesh_pretransformed.glb",
            "scale": root / "scale.json",
            "scale_registration_pose": root / "scale_registration_pose.json",
        }
        for key, live_path in fixed_paths.items():
            _match_lineage_file(
                chain[key], live_path, label=f"Upstream {track.name} {key}"
            )
        _validate_sam3d_transform_artifact(
            fixed_paths["sam3d_transform"], object_name=track.name
        )
        _validate_metric_scale_artifact(fixed_paths["scale"], object_name=track.name)
        load_object_to_camera_pose(fixed_paths["scale_registration_pose"])
        raw_poses = chain["raw_poses"]
        if not isinstance(raw_poses, list) or len(raw_poses) != geometry.frame_count:
            raise ContractError(
                f"Upstream lineage must bind every raw {track.name} pose"
            )
        raw_dir = root / "poses_raw"
        for frame_index, record in enumerate(raw_poses):
            raw_path = raw_dir / f"{frame_index:06d}.json"
            _match_lineage_file(
                record,
                raw_path,
                label=f"Upstream raw {track.name} pose {frame_index:06d}",
            )
            load_object_to_camera_pose(raw_path)

    stages = _exact_object(
        manifest["stages"],
        frozenset(("sam3d", "foundation_pose")),
        label="Upstream lineage stages",
    )
    sam3d = _validate_stage_identity(
        stages["sam3d"],
        label="SAM3D",
        repository=SAM3D_REPOSITORY,
        revision=SAM3D_REVISION,
        weight_identities=SAM3D_WEIGHT_IDENTITIES,
    )
    sam3d_parameters = _validate_sam3d_parameters(
        sam3d["parameters"], object_names=object_names, geometry=geometry
    )
    foundation_pose = _validate_stage_identity(
        stages["foundation_pose"],
        label="FoundationPose",
        repository=FOUNDATIONPOSE_REPOSITORY,
        revision=FOUNDATIONPOSE_REVISION,
        weight_identities=FOUNDATIONPOSE_WEIGHT_IDENTITIES,
    )
    _validate_foundationpose_parameters(
        foundation_pose["parameters"],
        object_names=object_names,
        geometry=geometry,
        sam3d_parameters=sam3d_parameters,
    )
    return UpstreamObjectLineage(
        path=path,
        manifest=manifest,
        record=file_record(path),
        verified_source_claim=json.loads(json.dumps(_VERIFIED_SOURCE_CLAIM)),
    )


def load_camera_model(
    path: str | Path, *, expected_width: int, expected_height: int
) -> CameraModel:
    """Load a strict V2D ``CameraIntrinsics`` JSON file."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = _read_json(path, label="camera intrinsics")
    if not isinstance(value, dict) or set(value) != _INTRINSIC_KEYS:
        raise ContractError(
            "Camera intrinsics must contain exactly fx, fy, cx, cy, width, and height"
        )
    width = _positive_integer(value["width"], label="Camera width")
    height = _positive_integer(value["height"], label="Camera height")
    if (width, height) != (expected_width, expected_height):
        raise ContractError(
            f"Camera resolution {(width, height)} does not match target "
            f"{(expected_width, expected_height)}"
        )
    fx = _numeric(value["fx"], label="Camera fx")
    fy = _numeric(value["fy"], label="Camera fy")
    cx = _numeric(value["cx"], label="Camera cx")
    cy = _numeric(value["cy"], label="Camera cy")
    if fx <= 0.0 or fy <= 0.0:
        raise ContractError("Camera fx and fy must be positive")
    if not 0.0 <= cx <= float(width) or not 0.0 <= cy <= float(height):
        raise ContractError("Camera principal point must lie within the target image")
    matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    matrix.setflags(write=False)
    return CameraModel(path=path, matrix=matrix, width=width, height=height)


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _validate_object_to_camera(matrix: np.ndarray, *, label: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ContractError(f"{label} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ContractError(f"{label} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0.0):
        raise ContractError(f"{label} is not a homogeneous transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-3, rtol=0.0):
        raise ContractError(f"{label} rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=2e-3, rtol=0.0):
        raise ContractError(f"{label} rotation is scaled or reflected")
    result = matrix.copy()
    result.setflags(write=False)
    return result


def load_object_to_camera_pose(path: str | Path) -> tuple[np.ndarray, str]:
    """Load one object-to-camera pose.

    Video2Data FoundationPose writes ``Transform3d`` dictionaries with a WXYZ
    quaternion, translation, and unit scale.  A literal nested 4x4 matrix is
    also accepted for portable exports.  Scale in a pose is rejected because
    this renderer requires metric meshes and SE(3) trajectories; accepting it
    would silently apply object scale twice.
    """

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = _read_json(path, label="object-to-camera pose")
    label = f"Object-to-camera pose {path}"
    if isinstance(value, list):
        try:
            matrix = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{label} matrix must be numeric") from exc
        pose_format = "matrix_4x4"
    elif isinstance(value, dict) and set(value) == _FOUNDATIONPOSE_KEYS:
        try:
            quaternion = np.asarray(value["rotation"], dtype=np.float64)
            translation = np.asarray(value["translation"], dtype=np.float64)
            scale = np.asarray(value["scale"], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{label} Transform3d fields must be numeric") from exc
        if quaternion.shape != (4,) or translation.shape != (3,) or scale.shape != (3,):
            raise ContractError(
                f"{label} Transform3d fields must have shapes (4,), (3,), and (3,)"
            )
        if not (
            np.isfinite(quaternion).all()
            and np.isfinite(translation).all()
            and np.isfinite(scale).all()
        ):
            raise ContractError(f"{label} Transform3d fields must be finite")
        norm = float(np.linalg.norm(quaternion))
        if not np.isclose(norm, 1.0, atol=1e-3, rtol=0.0):
            raise ContractError(f"{label} quaternion is not unit length")
        if not np.allclose(scale, 1.0, atol=1e-5, rtol=0.0):
            raise ContractError(
                f"{label} must have unit scale because the input mesh is metric"
            )
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _quaternion_wxyz_to_matrix(quaternion / norm)
        matrix[:3, 3] = translation
        pose_format = "foundationpose_transform3d_wxyz"
    else:
        raise ContractError(
            f"{label} must be a nested 4x4 matrix or an exact FoundationPose "
            "Transform3d object"
        )
    return _validate_object_to_camera(matrix, label=label), pose_format


def _pose_sequence(
    directory: Path, *, frame_count: int, object_name: str
) -> tuple[tuple[Path, ...], np.ndarray, str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"{object_name} poses directory: {directory}")
    expected = tuple(directory / f"{index:06d}.json" for index in range(frame_count))
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{object_name} poses are missing frames: {missing[:5]}"
        )
    actual = tuple(sorted(directory.glob("*.json")))
    if actual != expected:
        raise ContractError(
            f"{object_name} poses must contain exactly {frame_count} numbered JSON files"
        )
    matrices: list[np.ndarray] = []
    formats: list[str] = []
    for path in expected:
        matrix, pose_format = load_object_to_camera_pose(path)
        matrices.append(matrix)
        formats.append(pose_format)
    if len(set(formats)) != 1:
        raise ContractError(f"{object_name} pose sequence mixes JSON representations")
    stacked = np.stack(matrices, axis=0)
    stacked.setflags(write=False)
    return expected, stacked, formats[0]


def _validate_mesh_file(path: Path, *, object_name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{object_name} metric mesh: {path}")
    if path.suffix.lower() not in _MESH_SUFFIXES:
        raise ContractError(
            f"{object_name} mesh suffix must be one of {sorted(_MESH_SUFFIXES)}"
        )
    if path.stat().st_size <= 0:
        raise ContractError(f"{object_name} metric mesh is empty: {path}")
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ContractError(f"{object_name} metric mesh is a Git LFS pointer")


def _safe_sequence_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise ContractError("sequence_id must be one non-empty path segment")
    return value


def _safe_object_name(value: object) -> str:
    if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
        raise ContractError(
            "Object names must be lowercase identifiers containing letters, digits, "
            "and underscores"
        )
    return value


def _load_metric_object_tracks(
    object_specs: Sequence[tuple[str, str | Path, str | Path]],
    *,
    geometry: VideoGeometry,
) -> tuple[MetricObjectTrack, ...]:
    if not object_specs:
        raise ContractError("At least one metric object track is required")
    normalized_specs: list[tuple[str, Path, Path]] = []
    for spec in object_specs:
        if not isinstance(spec, (tuple, list)) or len(spec) != 3:
            raise ContractError(
                "Each object specification must contain NAME MESH POSES_DIR"
            )
        name = _safe_object_name(spec[0])
        normalized_specs.append(
            (
                name,
                Path(spec[1]).expanduser().resolve(),
                Path(spec[2]).expanduser().resolve(),
            )
        )
    names = [item[0] for item in normalized_specs]
    if len(set(names)) != len(names):
        raise ContractError("Object names must be unique")
    mesh_paths = [item[1] for item in normalized_specs]
    if len(set(mesh_paths)) != len(mesh_paths):
        raise ContractError("Every object must use a distinct metric mesh path")
    normalized_specs.sort(key=lambda item: item[0])

    tracks: list[MetricObjectTrack] = []
    for name, mesh_path, poses_dir in normalized_specs:
        _validate_mesh_file(mesh_path, object_name=name)
        pose_paths, poses, pose_format = _pose_sequence(
            poses_dir, frame_count=geometry.frame_count, object_name=name
        )
        tracks.append(
            MetricObjectTrack(
                name=name,
                mesh_path=mesh_path,
                poses_dir=poses_dir,
                pose_paths=pose_paths,
                object_to_camera=poses,
                pose_format=pose_format,
            )
        )
    return tuple(tracks)


def load_v2d_mesh_pose_inputs(
    *,
    sequence_id: str,
    source_video: str | Path,
    intrinsics_path: str | Path,
    object_specs: Sequence[tuple[str, str | Path, str | Path]],
    lineage_manifest: str | Path,
) -> V2DMeshPoseInputs:
    """Resolve and strictly validate all Video2Data renderer inputs."""

    sequence_id = _safe_sequence_id(sequence_id)
    source_video = Path(source_video).expanduser().resolve()
    geometry = probe_video(source_video)
    camera = load_camera_model(
        intrinsics_path,
        expected_width=geometry.width,
        expected_height=geometry.height,
    )
    tracks = _load_metric_object_tracks(object_specs, geometry=geometry)

    input_paths: dict[str, Path] = {
        "source_video": source_video,
        "camera_intrinsics": camera.path,
    }
    for track in tracks:
        input_paths[f"{track.name}_mesh"] = track.mesh_path
        for frame_index, pose_path in enumerate(track.pose_paths):
            input_paths[f"{track.name}_pose_{frame_index:06d}"] = pose_path

    upstream_lineage = load_upstream_object_lineage(
        lineage_manifest,
        sequence_id=sequence_id,
        source_video=source_video,
        geometry=geometry,
        camera=camera,
        objects=tracks,
    )
    input_paths["upstream_lineage"] = upstream_lineage.path

    repository_root = Path(__file__).resolve().parents[1]
    implementation_paths = [repository_root / path for path in IMPLEMENTATION_FILES]
    all_paths = [*input_paths.values(), *implementation_paths]
    signatures_before = {path: _stat_signature(path) for path in all_paths}
    provenance_inputs = {
        name: file_record(path) for name, path in sorted(input_paths.items())
    }
    # Fingerprinting must describe exactly the bytes whose semantics were
    # validated.  Re-read the small semantic inputs after hashing to close the
    # parse-before-fingerprint race; whitespace-only changes are harmless, but
    # any camera/pose/geometry change is refused.
    confirmed_geometry = probe_video(source_video)
    if confirmed_geometry != geometry:
        raise ContractError("Source video geometry changed during validation")
    confirmed_camera = load_camera_model(
        camera.path,
        expected_width=geometry.width,
        expected_height=geometry.height,
    )
    if not np.array_equal(confirmed_camera.matrix, camera.matrix):
        raise ContractError("Camera intrinsics changed during validation")
    for track in tracks:
        confirmed_paths, confirmed_poses, confirmed_format = _pose_sequence(
            track.poses_dir,
            frame_count=geometry.frame_count,
            object_name=track.name,
        )
        if (
            confirmed_paths != track.pose_paths
            or confirmed_format != track.pose_format
            or not np.array_equal(confirmed_poses, track.object_to_camera)
        ):
            raise ContractError(
                f"{track.name} object-to-camera poses changed during validation"
            )
    confirmed_lineage = load_upstream_object_lineage(
        upstream_lineage.path,
        sequence_id=sequence_id,
        source_video=source_video,
        geometry=geometry,
        camera=camera,
        objects=tracks,
    )
    if confirmed_lineage.record != upstream_lineage.record:
        raise ContractError("Upstream lineage manifest changed during validation")
    for path, signature in signatures_before.items():
        if _stat_signature(path) != signature:
            raise ContractError(f"Renderer input changed during validation: {path}")
    return V2DMeshPoseInputs(
        sequence_id=sequence_id,
        source_video=source_video,
        geometry=geometry,
        camera=camera,
        objects=tracks,
        upstream_lineage=upstream_lineage,
        provenance_inputs=provenance_inputs,
        input_signatures=signatures_before,
    )


def canonical_object_lineage_parameters(
    object_names: Sequence[str], *, reference_frame: int = 0
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return the audited parameter record for the sequence-105 V2D run.

    This is a manifest authoring helper, not an inference default.  Callers
    must change any value that differed in the actual upstream invocation;
    the resulting JSON is an immutable attestation of what ran.  The values
    match the canonical 105 experiment: depth-grounded SAM3D, the V2D
    FoundationPose scale/tracking settings, and the agreed anisotropic RTS
    smoothing parameters.
    """

    names = tuple(sorted(_safe_object_name(name) for name in object_names))
    if not names or len(set(names)) != len(names):
        raise ContractError(
            "Lineage parameter object names must be non-empty and unique"
        )
    _nonnegative_integer(reference_frame, label="Lineage reference_frame")
    sam3d: dict[str, dict[str, Any]] = {}
    foundation_pose: dict[str, dict[str, Any]] = {}
    for name in names:
        sam3d[name] = {
            "reference_frame": reference_frame,
            "seed": 105,
            "stage1_only": False,
            "with_mesh_postprocess": True,
            "with_texture_baking": False,
            "with_layout_postprocess": False,
            "use_vertex_color": True,
            "stage1_inference_steps": None,
            "geometry_grounding": "moge_depth",
        }
        foundation_pose[name] = {
            "scale_estimation": {
                "lo": 0.5,
                "hi": 2.0,
                "n_samples": 9,
                "n_levels": 4,
                "iou_weight": 1.0,
                "depth_weight": 1.0,
                "chamfer_weight": 0.0,
                "registration_iterations": 5,
            },
            "tracking": {
                "reference_frame": reference_frame,
                "target_width": None,
                "target_height": None,
                "reregister_iou_thresh": 0.3,
                "register_iteration": 10,
                "track_iteration": 5,
                "n_particles": 1,
                "particle_process_noise_t": 0.005,
                "particle_process_noise_r": 0.02,
                "particle_iteration": 3,
                "particle_mask_iou_weight": 1.0,
                "mask_depth": True,
            },
            "smoothing": {
                "process_noise_xy": 0.01,
                "process_noise_z": 0.01,
                "process_noise_r": 0.02,
                "measurement_noise_xy": 0.01,
                "measurement_noise_z": 0.04,
                "measurement_noise_r": 0.02,
                "min_iou": 0.1,
            },
        }
    return {"sam3d": sam3d, "foundation_pose": foundation_pose}


def sam3d_weight_paths(weights_dir: str | Path) -> dict[str, Path]:
    """Resolve the pinned SAM3D weight layout produced by ``download_weights``."""

    root = Path(weights_dir).expanduser().resolve()
    checkpoints = root / "hf-download" / "checkpoints"
    paths = {
        name: checkpoints / identity[0]
        for name, identity in SAM3D_WEIGHT_IDENTITIES.items()
        if name not in {"moge_v1", "dinov2_vitl14_reg4", "dinov2_vitb14_reg4"}
    }
    paths["moge_v1"] = (
        root
        / "hf_home"
        / "hub"
        / "models--Ruicheng--moge-vitl"
        / "snapshots"
        / SAM3D_MOGE_REVISION
        / "model.pt"
    )
    dino_root = root / "torch_home" / "hub" / "checkpoints"
    paths["dinov2_vitl14_reg4"] = dino_root / "dinov2_vitl14_reg4_pretrain.pth"
    paths["dinov2_vitb14_reg4"] = dino_root / "dinov2_vitb14_reg4_pretrain.pth"
    return paths


def foundationpose_weight_paths(weights_dir: str | Path) -> dict[str, Path]:
    """Resolve the pinned FoundationPose scorer/refiner weight layout."""

    root = Path(weights_dir).expanduser().resolve()
    scorer = root / "2024-01-11-20-02-45"
    refiner = root / "2023-10-28-18-33-37"
    return {
        "scorer_config": scorer / "config.yml",
        "scorer_model": scorer / "model_best.pth",
        "refiner_config": refiner / "config.yml",
        "refiner_model": refiner / "model_best.pth",
    }


def _verified_weight_records(
    paths: dict[str, Path],
    expected: dict[str, tuple[str, int, str]],
    *,
    label: str,
) -> list[dict[str, Any]]:
    if set(paths) != set(expected):
        raise ContractError(
            f"{label} weight paths must contain every pinned logical name"
        )
    result: list[dict[str, Any]] = []
    for name in sorted(expected):
        path = Path(os.path.abspath(str(Path(paths[name]).expanduser())))
        # Keep the stable logical cache path in the manifest.  Hugging Face
        # snapshots use symlinks into content-addressed blob directories; a
        # resolved record would lose the required checkpoint basename even
        # though the bytes are correct.
        record = file_record(path, recorded_path=str(path))
        basename, byte_count, digest = expected[name]
        if (
            path.name != basename
            or record["bytes"] != byte_count
            or record["sha256"] != digest
        ):
            raise ContractError(f"{label} weight {name} does not match its exact pin")
        result.append({"name": name, "record": record})
    return result


def write_upstream_object_lineage_manifest(
    manifest_path: str | Path,
    *,
    sequence_id: str,
    source_video: str | Path,
    intrinsics_path: str | Path,
    object_specs: Sequence[tuple[str, str | Path, str | Path]],
    moge_generation: str | Path,
    moge_intrinsics_dir: str | Path,
    stable_intrinsics_reproduction: str | Path,
    sam2_generation: str | Path,
    sam2_prompts: str | Path,
    sam2_object_ids: dict[str, int],
    sam3d_weights_dir: str | Path,
    foundationpose_weights_dir: str | Path,
    sam3d_container_image: str,
    sam3d_container_image_id: str,
    foundationpose_container_image: str,
    foundationpose_container_image_id: str,
    parameters: dict[str, Any],
    run_id: str | None = None,
) -> dict[str, Any]:
    """Fingerprint a completed RGB-only V2D object run into a strict manifest.

    Checkpoint files, RGB-only MoGe/SAM2 generation evidence, every SAM3D
    mesh/transform/scale artifact, and raw/smoothed FoundationPose trajectories
    are fingerprinted.  The output is then passed through the same live-input
    validator required by the renderer.
    """

    sequence_id = _safe_sequence_id(sequence_id)
    source_video = Path(source_video).expanduser().resolve()
    geometry = probe_video(source_video)
    camera = load_camera_model(
        intrinsics_path,
        expected_width=geometry.width,
        expected_height=geometry.height,
    )
    objects = _load_metric_object_tracks(object_specs, geometry=geometry)
    object_names = tuple(track.name for track in objects)
    if not isinstance(sam2_object_ids, dict) or set(sam2_object_ids) != set(
        object_names
    ):
        raise ContractError("sam2_object_ids must map exactly every rendered object")
    normalized_object_ids = {
        name: _positive_integer(
            sam2_object_ids[name], label=f"SAM2 object_id for {name}"
        )
        for name in object_names
    }
    if len(set(normalized_object_ids.values())) != len(normalized_object_ids):
        raise ContractError("sam2_object_ids values must be distinct")
    moge_generation = Path(moge_generation).expanduser().resolve()
    moge_intrinsics_dir = Path(moge_intrinsics_dir).expanduser().resolve()
    stable_intrinsics_reproduction = (
        Path(stable_intrinsics_reproduction).expanduser().resolve()
    )
    sam2_generation = Path(sam2_generation).expanduser().resolve()
    sam2_prompts = Path(sam2_prompts).expanduser().resolve()
    if not moge_intrinsics_dir.is_dir():
        raise FileNotFoundError(f"MoGe intrinsics directory: {moge_intrinsics_dir}")
    moge_intrinsics = tuple(
        moge_intrinsics_dir / f"{index:06d}.json"
        for index in range(geometry.frame_count)
    )
    if any(not path.is_file() for path in moge_intrinsics):
        raise FileNotFoundError("MoGe intrinsics evidence is incomplete")
    if tuple(sorted(moge_intrinsics_dir.glob("*.json"))) != moge_intrinsics:
        raise ContractError(
            "MoGe intrinsics evidence must contain exactly one JSON per frame"
        )
    parameters = _exact_object(
        parameters,
        frozenset(("sam3d", "foundation_pose")),
        label="Lineage parameters",
    )
    sam3d_stage = {
        "model": {"repository": SAM3D_REPOSITORY, "revision": SAM3D_REVISION},
        "container": {
            "image": sam3d_container_image,
            "image_id": sam3d_container_image_id,
        },
        "weights": _verified_weight_records(
            sam3d_weight_paths(sam3d_weights_dir),
            SAM3D_WEIGHT_IDENTITIES,
            label="SAM3D",
        ),
        "parameters": parameters["sam3d"],
    }
    foundationpose_stage = {
        "model": {
            "repository": FOUNDATIONPOSE_REPOSITORY,
            "revision": FOUNDATIONPOSE_REVISION,
        },
        "container": {
            "image": foundationpose_container_image,
            "image_id": foundationpose_container_image_id,
        },
        "weights": _verified_weight_records(
            foundationpose_weight_paths(foundationpose_weights_dir),
            FOUNDATIONPOSE_WEIGHT_IDENTITIES,
            label="FoundationPose",
        ),
        "parameters": parameters["foundation_pose"],
    }
    object_artifacts: list[dict[str, Any]] = []
    for track in objects:
        root = track.mesh_path.parent
        raw_poses = tuple(
            root / "poses_raw" / f"{index:06d}.json"
            for index in range(geometry.frame_count)
        )
        object_artifacts.append(
            {
                "name": track.name,
                "sam2_object_id": normalized_object_ids[track.name],
                "mesh": file_record(track.mesh_path),
                "poses": [file_record(path) for path in track.pose_paths],
                "chain": {
                    "sam3d_mesh": file_record(root / "sam3d_mesh.glb"),
                    "sam3d_transform": file_record(root / "sam3d_transform.json"),
                    "mesh_pretransformed": file_record(
                        root / "mesh_pretransformed.glb"
                    ),
                    "scale": file_record(root / "scale.json"),
                    "scale_registration_pose": file_record(
                        root / "scale_registration_pose.json"
                    ),
                    "raw_poses": [file_record(path) for path in raw_poses],
                },
            }
        )
    manifest = {
        "schema_version": UPSTREAM_LINEAGE_SCHEMA,
        "state": "complete",
        "run_id": run_id or str(uuid.uuid4()),
        "sequence_id": sequence_id,
        "geometry": geometry.as_dict(),
        "source_video": file_record(source_video),
        "rgb_evidence": {
            "moge_generation": file_record(moge_generation),
            "moge_intrinsics": [file_record(path) for path in moge_intrinsics],
            "stable_intrinsics_reproduction": file_record(
                stable_intrinsics_reproduction
            ),
            "sam2_generation": file_record(sam2_generation),
            "sam2_prompts": file_record(sam2_prompts),
        },
        "artifacts": {
            "intrinsics": file_record(camera.path),
            "objects": object_artifacts,
        },
        "stages": {
            "sam3d": sam3d_stage,
            "foundation_pose": foundationpose_stage,
        },
    }
    destination = Path(manifest_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"Upstream lineage manifest already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _validate_container_identity(sam3d_stage["container"], label="SAM3D")
    _validate_container_identity(
        foundationpose_stage["container"], label="FoundationPose"
    )
    try:
        _write_json_atomic(destination, manifest)
        load_upstream_object_lineage(
            destination,
            sequence_id=sequence_id,
            source_video=source_video,
            geometry=geometry,
            camera=camera,
            objects=objects,
        )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return manifest


def opencv_pose_to_pyrender_pose(object_to_camera: np.ndarray) -> np.ndarray:
    """Map an OpenCV object-to-camera SE(3) pose into pyrender scene space."""

    matrix = _validate_object_to_camera(
        object_to_camera, label="OpenCV object-to-camera pose"
    )
    # The pyrender camera is at identity.  This is a basis change on camera
    # coordinates only, not a conjugation of the object coordinate system.
    return CV_TO_OPENGL @ matrix


def metric_depth_layer(raw_depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Normalize a pyrender layer to float32 metric depth with ``+inf`` background."""

    value = np.asarray(raw_depth)
    if value.shape != shape:
        raise V2DMeshPoseOccluderError(
            f"pyrender depth shape {value.shape} does not match expected {shape}"
        )
    if not np.issubdtype(value.dtype, np.number):
        raise V2DMeshPoseOccluderError("pyrender depth must be numeric")
    value = value.astype(np.float32, copy=True)
    valid = np.isfinite(value) & (value > 0.0)
    value[~valid] = np.inf
    return value


def nearest_depth_union(
    layers: Sequence[np.ndarray], *, shape: tuple[int, int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest positive finite depth and its boolean validity mask."""

    if not layers:
        raise ContractError("At least one object depth layer is required")
    inferred_shape = np.asarray(layers[0]).shape if shape is None else shape
    if len(inferred_shape) != 2:
        raise ContractError("Object depth layers must be two-dimensional")
    union = np.full(inferred_shape, np.inf, dtype=np.float32)
    for index, layer in enumerate(layers):
        normalized = metric_depth_layer(layer, inferred_shape)
        if np.any(np.isfinite(normalized) & (normalized <= 0.0)):
            raise ContractError(f"Object depth layer {index} contains invalid depth")
        np.minimum(union, normalized, out=union)
    mask = np.isfinite(union)
    return mask, union


class PyrenderDepthLayerRenderer:
    """GPU-backed depth renderer; imports OpenGL dependencies lazily."""

    def __init__(self, inputs: V2DMeshPoseInputs, *, opengl_platform: str) -> None:
        if opengl_platform not in {"egl", "osmesa"}:
            raise ContractError("OpenGL platform must be 'egl' or 'osmesa'")
        existing = os.environ.get("PYOPENGL_PLATFORM")
        if existing is not None and existing != opengl_platform:
            raise V2DMeshPoseOccluderError(
                f"PYOPENGL_PLATFORM is already {existing!r}, not {opengl_platform!r}"
            )
        os.environ["PYOPENGL_PLATFORM"] = opengl_platform
        try:
            import pyrender
            import trimesh
        except ImportError as exc:
            raise V2DMeshPoseOccluderError(
                "pyrender and trimesh are required for mesh/pose occluder rendering"
            ) from exc

        self._pyrender = pyrender
        self._renderer = None
        self._scenes: list[Any] = []
        self._nodes: list[Any] = []
        try:
            self._renderer = pyrender.OffscreenRenderer(
                viewport_width=inputs.geometry.width,
                viewport_height=inputs.geometry.height,
            )
            camera = inputs.camera.matrix
            for track in inputs.objects:
                mesh = trimesh.load(track.mesh_path, force="mesh", process=False)
                if not isinstance(mesh, trimesh.Trimesh):
                    raise V2DMeshPoseOccluderError(
                        f"Expected one triangle mesh for {track.name}: {track.mesh_path}"
                    )
                vertices = np.asarray(mesh.vertices, dtype=np.float64)
                faces = np.asarray(mesh.faces)
                if (
                    vertices.ndim != 2
                    or vertices.shape[1:] != (3,)
                    or vertices.shape[0] < 3
                    or faces.ndim != 2
                    or faces.shape[1:] != (3,)
                    or faces.shape[0] < 1
                    or not np.isfinite(vertices).all()
                ):
                    raise V2DMeshPoseOccluderError(
                        f"Invalid triangle mesh for {track.name}: {track.mesh_path}"
                    )
                radius = float(
                    np.max(np.linalg.norm(vertices - vertices.mean(axis=0), axis=1))
                )
                if not 1e-4 <= radius <= 5.0:
                    raise V2DMeshPoseOccluderError(
                        f"Metric mesh radius {radius:.6g} m is implausible for "
                        f"{track.name}"
                    )
                scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
                scene.add(
                    pyrender.IntrinsicsCamera(
                        fx=float(camera[0, 0]),
                        fy=float(camera[1, 1]),
                        cx=float(camera[0, 2]),
                        cy=float(camera[1, 2]),
                        znear=ZNEAR_METRES,
                        zfar=ZFAR_METRES,
                    ),
                    pose=np.eye(4),
                )
                node = scene.add(
                    pyrender.Mesh.from_trimesh(mesh, smooth=False),
                    pose=np.eye(4),
                    name=track.name,
                )
                self._scenes.append(scene)
                self._nodes.append(node)
        except Exception:
            self.close()
            raise

    def render(self, object_index: int, object_to_camera: np.ndarray) -> np.ndarray:
        scene = self._scenes[object_index]
        scene.set_pose(
            self._nodes[object_index],
            pose=opencv_pose_to_pyrender_pose(object_to_camera),
        )
        flags = (
            self._pyrender.RenderFlags.DEPTH_ONLY
            | self._pyrender.RenderFlags.SKIP_CULL_FACES
        )
        return np.asarray(self._renderer.render(scene, flags=flags))

    def close(self) -> None:
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            renderer.delete()
            self._renderer = None


def _validate_execution_identity(
    *, renderer_image: str, renderer_image_id: str, opengl_platform: str
) -> None:
    if not isinstance(renderer_image, str) or not renderer_image.strip():
        raise ContractError("renderer_image must be a non-empty image reference")
    if (
        not isinstance(renderer_image_id, str)
        or _IMAGE_ID.fullmatch(renderer_image_id) is None
    ):
        raise ContractError(
            "renderer_image_id must be an immutable sha256 Docker image ID"
        )
    if opengl_platform not in {"egl", "osmesa"}:
        raise ContractError("OpenGL platform must be 'egl' or 'osmesa'")


def render_v2d_mesh_pose_occluder(
    inputs: V2DMeshPoseInputs,
    output_dir: str | Path,
    *,
    renderer_image: str,
    renderer_image_id: str,
    host_output_dir: str | Path | None = None,
    opengl_platform: str = "egl",
    renderer_factory: Callable[[V2DMeshPoseInputs, str], DepthLayerRenderer]
    | None = None,
) -> dict[str, Any]:
    """Render and atomically commit a generic mesh/pose occluder bundle."""

    if not isinstance(inputs, V2DMeshPoseInputs):
        raise TypeError("inputs must be V2DMeshPoseInputs")
    _validate_execution_identity(
        renderer_image=renderer_image,
        renderer_image_id=renderer_image_id,
        opengl_platform=opengl_platform,
    )
    confirmed_lineage = load_upstream_object_lineage(
        inputs.upstream_lineage.path,
        sequence_id=inputs.sequence_id,
        source_video=inputs.source_video,
        geometry=inputs.geometry,
        camera=inputs.camera,
        objects=inputs.objects,
    )
    if confirmed_lineage.record != inputs.upstream_lineage.record:
        raise ContractError("Upstream lineage manifest changed after input validation")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Occluder output already exists: {destination}")
    if host_output_dir is None:
        published_destination = destination
    else:
        published_destination = Path(host_output_dir)
        if not published_destination.is_absolute():
            raise ContractError("host_output_dir must be an absolute host path")
        published_destination = Path(os.path.normpath(str(published_destination)))
    destination.parent.mkdir(parents=True, exist_ok=True)

    for path, signature in inputs.input_signatures.items():
        if _stat_signature(path) != signature:
            raise ContractError(f"Renderer input changed after validation: {path}")

    repository_root = Path(__file__).resolve().parents[1]
    implementation_paths = [repository_root / path for path in IMPLEMENTATION_FILES]
    implementation_records = [
        file_record(path, recorded_path=path.relative_to(repository_root).as_posix())
        for path in implementation_paths
    ]
    run_id = str(uuid.uuid4())
    staging = destination.parent / f".{destination.name}.{run_id}.partial"
    staging.mkdir()
    mask_path = staging / OCCLUDER_ARTIFACT_NAMES["mask"]
    depth_path = staging / OCCLUDER_ARTIFACT_NAMES["depth"]
    mask_memmap = None
    depth_memmap = None
    renderer: DepthLayerRenderer | None = None
    committed = False
    try:
        shape = (
            inputs.geometry.frame_count,
            inputs.geometry.height,
            inputs.geometry.width,
        )
        mask_memmap = np.lib.format.open_memmap(
            mask_path, mode="w+", dtype=np.bool_, shape=shape
        )
        depth_memmap = np.lib.format.open_memmap(
            depth_path, mode="w+", dtype=np.float32, shape=shape
        )
        renderer = (
            renderer_factory(inputs, opengl_platform)
            if renderer_factory is not None
            else PyrenderDepthLayerRenderer(inputs, opengl_platform=opengl_platform)
        )
        per_object_pixels = {track.name: [] for track in inputs.objects}
        union_pixels: list[int] = []
        frame_shape = shape[1:]
        for frame_index in range(inputs.geometry.frame_count):
            layers: list[np.ndarray] = []
            for object_index, track in enumerate(inputs.objects):
                layer = metric_depth_layer(
                    renderer.render(object_index, track.object_to_camera[frame_index]),
                    frame_shape,
                )
                layers.append(layer)
                per_object_pixels[track.name].append(int(np.isfinite(layer).sum()))
            frame_mask, frame_depth = nearest_depth_union(layers, shape=frame_shape)
            mask_memmap[frame_index] = frame_mask
            depth_memmap[frame_index] = frame_depth
            union_pixels.append(int(frame_mask.sum()))

        minimum_pixels = max(
            16,
            int(np.ceil(inputs.geometry.width * inputs.geometry.height * 1e-6)),
        )
        required_visible_frames = max(
            1, int(np.ceil(inputs.geometry.frame_count * 0.10))
        )
        object_statistics: dict[str, dict[str, Any]] = {}
        for track in inputs.objects:
            counts = per_object_pixels[track.name]
            visible_frames = int(np.count_nonzero(np.asarray(counts) >= minimum_pixels))
            if visible_frames < required_visible_frames:
                raise V2DMeshPoseOccluderError(
                    f"{track.name} render is blank or nearly blank in too many frames; "
                    "recheck mesh scale, pose direction, and camera convention"
                )
            object_statistics[track.name] = {
                "total_pixels": int(sum(counts)),
                "min_pixels_per_frame": int(min(counts)),
                "max_pixels_per_frame": int(max(counts)),
                "visible_frame_count": visible_frames,
            }

        mask_memmap.flush()
        depth_memmap.flush()
        validate_mask_array(mask_memmap, inputs.geometry)
        validate_depth_array(
            depth_memmap, mask_memmap, inputs.geometry, name="V2D object depth"
        )
        del mask_memmap
        del depth_memmap
        mask_memmap = None
        depth_memmap = None
        renderer.close()
        renderer = None

        for path, signature in inputs.input_signatures.items():
            if _stat_signature(path) != signature:
                raise ContractError(f"Renderer input changed during generation: {path}")
        current_implementation_records = [
            file_record(
                path, recorded_path=path.relative_to(repository_root).as_posix()
            )
            for path in implementation_paths
        ]
        if current_implementation_records != implementation_records:
            raise ContractError("Renderer implementation changed during generation")

        metadata = {
            "schema_version": OCCLUDER_DEPTH_SCHEMA,
            "state": "complete",
            "run_id": run_id,
            "sequence_id": inputs.sequence_id,
            "completed_at": _utc_now(),
            # This value is temporarily the execution path so the exact bundle
            # can be validated inside its renderer container.  If the bind
            # mount has a different host path, it is atomically rewritten to
            # ``published_destination`` only after that validation succeeds.
            "host_output_dir": str(destination),
            "geometry": inputs.geometry.as_dict(),
            "occluder_scope": "tool_and_target",
            "source_modalities": ["rgb"],
            "producer": {
                "name": PRODUCER_NAME,
                "method": (
                    "metric SAM3D meshes tracked as OpenCV object-to-camera SE(3) "
                    "by FoundationPose and rasterized independently with pyrender"
                ),
                "version": PRODUCER_VERSION,
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
                "inputs": inputs.provenance_inputs,
                "implementation_sources": implementation_records,
            },
            "estimation": {
                "uses_ground_truth": confirmed_lineage.verified_source_claim[
                    "uses_ground_truth"
                ],
                "upstream_input_modalities": confirmed_lineage.verified_source_claim[
                    "primary_input_modalities"
                ],
                "camera_intrinsics_source": confirmed_lineage.verified_source_claim[
                    "camera_intrinsics_source"
                ],
                "object_mesh_source": confirmed_lineage.verified_source_claim[
                    "object_mesh_source"
                ],
                "object_pose_source": confirmed_lineage.verified_source_claim[
                    "object_pose_source"
                ],
                "camera_intrinsics_policy": (
                    "legacy stable K consumed upstream; its numeric values were "
                    "independently reproduced from per-frame RGB-estimated MoGe "
                    "intrinsics"
                ),
                "upstream_lineage": {
                    "schema_version": confirmed_lineage.manifest["schema_version"],
                    "run_id": confirmed_lineage.manifest["run_id"],
                    "manifest": confirmed_lineage.record,
                    "stages": confirmed_lineage.manifest["stages"],
                },
            },
            "render": {
                "renderer_image": renderer_image,
                "renderer_image_id": renderer_image_id,
                "opengl_platform": opengl_platform,
                "znear_metres": ZNEAR_METRES,
                "zfar_metres": ZFAR_METRES,
                "layer_composition": "nearest_positive_finite_camera_z",
                "coordinate_conventions": {
                    "input_pose": "T_camera_object (OpenCV object-to-camera SE(3))",
                    "camera": "OpenCV +x right, +y down, +z forward",
                    "pyrender_camera": "OpenGL +x right, +y up, -z forward",
                    "cv_to_opengl": CV_TO_OPENGL.tolist(),
                    "depth": "positive metric OpenCV camera-z; invalid +inf",
                },
                "intrinsic": inputs.camera.matrix.tolist(),
                "objects": [
                    {
                        "name": track.name,
                        "mesh_path": str(track.mesh_path),
                        "mesh_units": "metres",
                        "poses_dir": str(track.poses_dir),
                        "pose_format": track.pose_format,
                        "pose_count": len(track.pose_paths),
                    }
                    for track in inputs.objects
                ],
            },
            "statistics": {
                "total_occluder_pixels": int(sum(union_pixels)),
                "min_occluder_pixels_per_frame": int(min(union_pixels)),
                "max_occluder_pixels_per_frame": int(max(union_pixels)),
                "visibility_pixel_threshold": minimum_pixels,
                "required_visible_frame_count": required_visible_frames,
                "objects": object_statistics,
            },
        }
        _write_json(staging / OCCLUDER_METADATA_NAME, metadata)
        staging.replace(destination)
        committed = True
        validate_occluder_depth_bundle(
            destination / OCCLUDER_METADATA_NAME,
            inputs.geometry,
            verify_provenance_files=True,
            provenance_root=repository_root,
        )
        if published_destination != destination:
            metadata["host_output_dir"] = str(published_destination)
            _write_json_atomic(destination / OCCLUDER_METADATA_NAME, metadata)
        return metadata
    except Exception:
        if mask_memmap is not None:
            mask_memmap.flush()
        if depth_memmap is not None:
            depth_memmap.flush()
        if renderer is not None:
            renderer.close()
        if staging.exists():
            shutil.rmtree(staging)
        if committed and destination.exists():
            shutil.rmtree(destination)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument(
        "--lineage-manifest",
        required=True,
        type=Path,
        help="immutable manifest written by write_upstream_object_lineage_manifest",
    )
    parser.add_argument(
        "--object",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "METRIC_MESH", "POSES_DIR"),
        help="repeat once per task object",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--host-output-dir",
        type=Path,
        help=(
            "absolute host path of output-dir when rendering through a differently "
            "named container bind mount"
        ),
    )
    parser.add_argument("--renderer-image", required=True)
    parser.add_argument("--renderer-image-id", required=True)
    parser.add_argument("--opengl-platform", choices=("egl", "osmesa"), default="egl")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = load_v2d_mesh_pose_inputs(
        sequence_id=args.sequence_id,
        source_video=args.source_video,
        intrinsics_path=args.intrinsics,
        object_specs=tuple(tuple(spec) for spec in args.object),
        lineage_manifest=args.lineage_manifest,
    )
    _validate_execution_identity(
        renderer_image=args.renderer_image,
        renderer_image_id=args.renderer_image_id,
        opengl_platform=args.opengl_platform,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "state": "validated",
                    "sequence_id": inputs.sequence_id,
                    "geometry": inputs.geometry.as_dict(),
                    "objects": [
                        {
                            "name": track.name,
                            "mesh_path": str(track.mesh_path),
                            "poses_dir": str(track.poses_dir),
                            "pose_format": track.pose_format,
                        }
                        for track in inputs.objects
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    metadata = render_v2d_mesh_pose_occluder(
        inputs,
        args.output_dir,
        renderer_image=args.renderer_image,
        renderer_image_id=args.renderer_image_id,
        host_output_dir=args.host_output_dir,
        opengl_platform=args.opengl_platform,
    )
    print(
        f"Wrote V2D mesh/pose occluder {metadata['run_id']} -> "
        f"{metadata['host_output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
