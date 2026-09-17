"""Validate one complete MV-HOI export before it is committed or published."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import av
import h5py
import numpy as np
import torch
import trimesh
import yaml

from v2d.common.ffv1_sidecar import read_ffv1_metadata, verify_ffv1_sidecar
from v2d.common.video import FrameSource
try:
    from .interaction_trim import (
        MHR_PARAMETER_TEMPORAL_KEYS,
        SOMA_REQUIRED_TEMPORAL_KEYS,
        SOMA_TEMPORAL_KEYS,
        validate_trim_manifest,
    )
except ImportError:
    from interaction_trim import (
        MHR_PARAMETER_TEMPORAL_KEYS,
        SOMA_REQUIRED_TEMPORAL_KEYS,
        SOMA_TEMPORAL_KEYS,
        validate_trim_manifest,
    )


LEFT_CAMERAS = (
    "back_stereo_camera_left",
    "front_stereo_camera_left",
    "left_stereo_camera_left",
    "right_stereo_camera_left",
)
RIGHT_CAMERAS = tuple(camera.replace("_left", "_right") for camera in LEFT_CAMERAS)
RGB_CAMERAS = LEFT_CAMERAS + RIGHT_CAMERAS


def _require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Missing or empty required export file: {path}")
    return path


def _validate_soma_temporal_arrays(
    soma: Mapping[str, Any], frame_count: int,
) -> None:
    """Validate required legacy arrays and any optional temporal extensions."""
    keys = set(soma)
    missing = SOMA_REQUIRED_TEMPORAL_KEYS - keys
    if missing:
        raise ValueError(
            f"soma_params.npz is missing temporal keys: {sorted(missing)}"
        )
    for key in sorted(SOMA_TEMPORAL_KEYS & keys):
        value = soma[key]
        if value.ndim < 1 or int(value.shape[0]) != frame_count:
            raise ValueError(
                f"soma_params.npz temporal key {key} differs from RGB"
            )


def _exact_camera_files(directory: Path, suffix: str, cameras: tuple[str, ...]) -> dict[str, Path]:
    if not directory.is_dir():
        raise ValueError(f"Missing required export directory: {directory}")
    expected = {f"{camera}{suffix}" for camera in cameras}
    actual = {path.name for path in directory.iterdir() if path.is_file() and path.suffix == suffix}
    if actual != expected:
        raise ValueError(
            f"Invalid camera set under {directory}: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    return {camera: directory / f"{camera}{suffix}" for camera in cameras}


def _ffv1_group(
    root: Path,
    directory: Path,
    cameras: tuple[str, ...],
    kind: str,
    known_payload_sha256: Mapping[str, str] | None = None,
) -> dict[str, dict]:
    metadata_paths = _exact_camera_files(directory, ".h5", cameras)
    records: dict[str, dict] = {}
    referenced_sidecars: set[str] = set()
    for camera, path in metadata_paths.items():
        info = read_ffv1_metadata(path)
        if info["kind"] != kind:
            raise ValueError(f"{path} is {info['kind']}, expected {kind}")
        if known_payload_sha256 is None:
            verify_ffv1_sidecar(path, verify_decoded_frames=False)
        else:
            relative_sidecar = Path(info["sidecar_path"]).relative_to(root).as_posix()
            actual_sha256 = known_payload_sha256.get(relative_sidecar)
            if actual_sha256 != info["sidecar_sha256"]:
                raise ValueError(
                    f"CARI4D FFV1 sidecar SHA-256 does not match metadata: {path}"
                )
        referenced_sidecars.add(info["sidecar_basename"])
        records[camera] = info
    actual_sidecars = {path.name for path in directory.glob("*.mkv")}
    if actual_sidecars != referenced_sidecars:
        raise ValueError(
            f"Invalid FFV1 sidecar set under {directory}: "
            f"missing={sorted(referenced_sidecars - actual_sidecars)}, "
            f"unexpected={sorted(actual_sidecars - referenced_sidecars)}"
        )
    unexpected = [
        path.name for path in directory.iterdir()
        if path.is_file() and path.suffix not in {".h5", ".mkv"}
    ]
    if unexpected:
        raise ValueError(f"Unexpected FFV1 group files under {directory}: {sorted(unexpected)}")
    return records


def _h5_group(directory: Path, cameras: tuple[str, ...]) -> dict[str, dict]:
    paths = _exact_camera_files(directory, ".h5", cameras)
    records: dict[str, dict] = {}
    for camera, path in paths.items():
        with FrameSource.from_path(path) as source:
            if source.n_frames <= 0:
                raise ValueError(f"Empty frame archive: {path}")
            records[camera] = {
                "n_frames": source.n_frames,
                "stems": list(source.stems),
                "dtype": str(source.dtype),
                "image_size": tuple(source.image_size),
            }
    unexpected = [
        path.name for path in directory.iterdir()
        if path.is_file() and path.suffix != ".h5"
    ]
    if unexpected:
        raise ValueError(f"Unexpected mask files under {directory}: {sorted(unexpected)}")
    return records


def _mask_group(directory: Path, cameras: tuple[str, ...]) -> dict[str, dict]:
    """Validate packed-H5 or legacy per-camera PNG mask archives."""
    if not directory.is_dir():
        raise ValueError(f"Missing required export directory: {directory}")

    h5_paths = list(directory.glob("*.h5"))
    camera_directories = {path.name: path for path in directory.iterdir() if path.is_dir()}
    if h5_paths:
        if camera_directories:
            raise ValueError(f"Mixed H5 and PNG-directory mask layouts under {directory}")
        return _h5_group(directory, cameras)

    expected = set(cameras)
    actual = set(camera_directories)
    root_files = sorted(path.name for path in directory.iterdir() if path.is_file())
    if actual != expected or root_files:
        raise ValueError(
            f"Invalid mask camera directories under {directory}: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}, "
            f"root_files={root_files}"
        )

    records: dict[str, dict] = {}
    for camera in cameras:
        camera_dir = camera_directories[camera]
        unexpected = sorted(
            path.name
            for path in camera_dir.iterdir()
            if not path.is_file() or path.suffix.lower() != ".png"
        )
        if unexpected:
            raise ValueError(
                f"Unexpected files in mask camera directory {camera_dir}: {unexpected}"
            )
        with FrameSource.from_path(camera_dir) as source:
            if source.n_frames <= 0:
                raise ValueError(f"Empty mask camera directory: {camera_dir}")
            records[camera] = {
                "n_frames": source.n_frames,
                "stems": list(source.stems),
            }
    return records


def _video_frame_count(path: Path) -> int:
    _require_file(path)
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"Expected exactly one video stream: {path}")
        stream = container.streams.video[0]
        declared_count = int(stream.frames or 0)
        if declared_count > 0:
            if next(container.decode(stream), None) is None:
                raise ValueError(f"Video has no decodable frames: {path}")
            return declared_count
        count = sum(1 for _ in container.decode(stream))
    if count <= 0:
        raise ValueError(f"Video has no decodable frames: {path}")
    return count


def _same_frame_identity(label: str, records: dict[str, dict]) -> tuple[int, list[str]]:
    identities = {
        (int(record["n_frames"]), tuple(record["stems"]))
        for record in records.values()
    }
    if len(identities) != 1:
        raise ValueError(f"{label} camera frame counts/stems are not balanced")
    count, stems = next(iter(identities))
    return count, list(stems)


def _validate_edex(path: Path, *, frame_count: int) -> dict:
    payload = json.loads(_require_file(path).read_text())
    header = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(header, dict):
        raise ValueError("EDEX must contain a JSON object header")
    cameras = header.get("cameras") or header.get("header", {}).get("cameras")
    if not isinstance(cameras, list) or len(cameras) < len(RGB_CAMERAS):
        raise ValueError("EDEX does not contain all eight camera calibrations")
    if int(header.get("frame_start", -1)) != 0:
        raise ValueError("Exported EDEX frame_start must be zero")
    if int(header.get("frame_end", -1)) != frame_count:
        raise ValueError("Exported EDEX frame_end differs from RGB")
    return {
        "camera_count": len(cameras),
        "frame_start": 0,
        "frame_end": frame_count,
    }


def _load_metadata(path: Path) -> dict[str, Any]:
    """Load YAML metadata without requiring its scalar values to be JSON-native."""
    metadata = yaml.safe_load(_require_file(path).read_text())
    if not isinstance(metadata, dict):
        raise ValueError("hoi_metadata.yaml must contain a mapping")
    return metadata


def _payload_sha256(
    root: Path,
    path: Path,
    known_payload_sha256: Mapping[str, str] | None,
) -> str:
    if known_payload_sha256 is None:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    relative = path.relative_to(root).as_posix()
    try:
        return known_payload_sha256[relative]
    except KeyError as exc:
        raise ValueError(f"Missing precomputed payload SHA-256: {relative}") from exc


def _validate_non_camera_assets(
    root: Path,
    frame_count: int,
    trim_manifest: dict[str, Any],
    known_payload_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    metadata = _load_metadata(root / "hoi_metadata.yaml")
    if int(metadata.get("frame_count", -1)) != frame_count:
        raise ValueError("hoi_metadata.yaml frame_count differs from RGB")
    metadata_trim = metadata.get("interaction_trim")
    if (
        not isinstance(metadata_trim, dict)
        or metadata_trim.get("decision_sha256")
        != trim_manifest.get("decision_sha256")
    ):
        raise ValueError("hoi_metadata.yaml interaction trim provenance differs")
    edex = _validate_edex(root / "edex", frame_count=frame_count)

    mesh_path = _require_file(root / "object_mesh" / "output_aligned.glb")
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Aligned object mesh is empty")
    symmetry_path = _require_file(root / "object_mesh" / "output_symmetry.json")
    symmetry = json.loads(symmetry_path.read_text())
    if not isinstance(symmetry, dict):
        raise ValueError("Object symmetry metadata must be a JSON object")
    alignment = symmetry.get("alignment")
    if not isinstance(alignment, dict):
        raise ValueError("Object symmetry metadata is missing alignment")
    centroid = np.asarray(alignment.get("centroid"), dtype=float)
    rotation = np.asarray(alignment.get("rotation"), dtype=float)
    if centroid.shape != (3,) or rotation.size != 16:
        raise ValueError("Object symmetry alignment has invalid centroid/rotation")
    rotation = rotation.reshape(4, 4)
    if not np.isfinite(centroid).all() or not np.isfinite(rotation).all():
        raise ValueError("Object symmetry alignment contains non-finite values")
    if not np.allclose(rotation[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Object symmetry alignment is not homogeneous")
    if not np.allclose(rotation[:3, :3].T @ rotation[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("Object symmetry alignment rotation is not orthonormal")
    if not isinstance(symmetry.get("symmetries_discrete"), list) or not isinstance(
        symmetry.get("symmetries_continuous"), list
    ):
        raise ValueError("Object symmetry metadata is missing symmetry lists")

    pose_path = _require_file(root / "poses.npy")
    poses = np.load(pose_path, allow_pickle=False)
    if poses.ndim != 3 or poses.shape[0] != frame_count or poses.shape[-2:] != (4, 4):
        raise ValueError(f"poses.npy has incompatible shape {poses.shape}")
    valid_path = root / "pose_valid_mask.npy"
    if valid_path.exists():
        valid = np.load(_require_file(valid_path), allow_pickle=False)
        if valid.shape != (frame_count,):
            raise ValueError(f"pose_valid_mask.npy has incompatible shape {valid.shape}")

    mhr_params = torch.load(
        _require_file(root / "mhr_params_mv.pt"), weights_only=False, map_location="cpu",
    )
    if not isinstance(mhr_params, dict):
        raise ValueError("mhr_params_mv.pt must contain a mapping")
    missing_mhr_keys = MHR_PARAMETER_TEMPORAL_KEYS - set(mhr_params)
    if missing_mhr_keys:
        raise ValueError(
            f"mhr_params_mv.pt is missing temporal keys: {sorted(missing_mhr_keys)}"
        )
    for key in sorted(MHR_PARAMETER_TEMPORAL_KEYS):
        value = mhr_params[key]
        if not hasattr(value, "shape") or int(value.shape[0]) != frame_count:
            raise ValueError(f"mhr_params_mv.pt temporal key {key} differs from RGB")
    mhr_mesh = torch.load(
        _require_file(root / "mhr_mesh_mv.pt"), weights_only=False, map_location="cpu",
    )
    if not isinstance(mhr_mesh, dict) or not {"pred_vertices", "faces"}.issubset(mhr_mesh):
        raise ValueError("mhr_mesh_mv.pt is missing pred_vertices/faces")
    if int(mhr_mesh["pred_vertices"].shape[0]) != frame_count:
        raise ValueError("mhr_mesh_mv.pt frame count differs from RGB")

    with np.load(
        _require_file(root / "soma_params.npz"), allow_pickle=False,
    ) as soma:
        if not soma.files:
            raise ValueError("soma_params.npz contains no arrays")
        _validate_soma_temporal_arrays(soma, frame_count)
    ground = json.loads(_require_file(root / "ground_plane.json").read_text())
    if not isinstance(ground, dict) or not (
        "plane" in ground or {"normal", "offset"}.issubset(ground)
    ):
        raise ValueError("ground_plane.json is missing plane coefficients")
    overlay_frames = _video_frame_count(root / "tiled_hoi_overlay.mp4")
    if overlay_frames != frame_count:
        raise ValueError("Tiled overlay frame count differs from RGB")
    return {
        "edex": edex,
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_faces": int(len(mesh.faces)),
        "object_pose_frame": "aligned",
        "pose_sha256": _payload_sha256(
            root, pose_path, known_payload_sha256,
        ),
        "mesh_sha256": _payload_sha256(
            root, mesh_path, known_payload_sha256,
        ),
        "symmetry_sha256": _payload_sha256(
            root, symmetry_path, known_payload_sha256,
        ),
        "overlay_frames": overlay_frames,
        "metadata_frame_count": int(metadata["frame_count"]),
    }


def validate_export_payload(
    root: str | Path,
    *,
    known_payload_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate every required export asset and cross-modality frame identity."""
    root = Path(root)
    images = _ffv1_group(
        root, root / "images", RGB_CAMERAS, "rgb", known_payload_sha256,
    )
    anonymized = _ffv1_group(
        root,
        root / "images_anonymized",
        RGB_CAMERAS,
        "rgb",
        known_payload_sha256,
    )
    depth = _ffv1_group(
        root, root / "depth", LEFT_CAMERAS, "depth", known_payload_sha256,
    )
    object_masks = _mask_group(root / "object_masks", LEFT_CAMERAS)
    human_masks = _mask_group(root / "human_masks", LEFT_CAMERAS)

    frame_count, stems = _same_frame_identity("images", images)
    expected_stems = [f"{index:06d}" for index in range(frame_count)]
    if stems != expected_stems:
        raise ValueError("Exported frame stems must be reindexed from zero")
    for label, records in (
        ("images_anonymized", anonymized),
        ("depth", depth),
        ("object_masks", object_masks),
        ("human_masks", human_masks),
    ):
        other_count, other_stems = _same_frame_identity(label, records)
        if other_count != frame_count or other_stems != stems:
            raise ValueError(f"{label} frame identity differs from original RGB")

    video_counts: dict[str, dict[str, int]] = {}
    for directory_name in ("videos", "videos_anonymized"):
        paths = _exact_camera_files(root / directory_name, ".mp4", RGB_CAMERAS)
        counts = {camera: _video_frame_count(path) for camera, path in paths.items()}
        if set(counts.values()) != {frame_count}:
            raise ValueError(f"{directory_name} frame counts differ from RGB: {counts}")
        video_counts[directory_name] = counts

    trim_manifest = json.loads(
        _require_file(root / "interaction_trim.json").read_text()
    )
    _, _, trim_frame_count = validate_trim_manifest(trim_manifest)
    if trim_frame_count != frame_count:
        raise ValueError("Interaction trim frame count differs from RGB")

    failure_segments_path = root / "failure_segments.json"
    if failure_segments_path.exists():
        failure_segments = json.loads(_require_file(failure_segments_path).read_text())
        if not isinstance(failure_segments, list):
            raise ValueError("failure_segments.json must contain a list")
        for segment in failure_segments:
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            if not 0 <= start < end <= frame_count:
                raise ValueError(
                    f"Failure segment is outside exported timeline: {segment}"
                )

    non_camera = _validate_non_camera_assets(
        root, frame_count, trim_manifest, known_payload_sha256,
    )
    return {
        "schema": "v2d.mv_hoi.export_validation.v2",
        "frame_count": frame_count,
        "interaction_trim_decision_sha256": trim_manifest["decision_sha256"],
        "stems_sha256": __import__("hashlib").sha256(
            json.dumps(stems, separators=(",", ":")).encode()
        ).hexdigest(),
        "camera_counts": {
            "images": len(images), "videos": len(video_counts["videos"]),
            "images_anonymized": len(anonymized),
            "videos_anonymized": len(video_counts["videos_anonymized"]),
            "depth": len(depth), "object_masks": len(object_masks),
            "human_masks": len(human_masks),
        },
        "non_camera": non_camera,
    }
