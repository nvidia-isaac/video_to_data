"""Verify and atomically add SHA-256 provenance to a completed v1 render bundle.

This is intentionally a metadata-only migration: it never invokes IK, OpenGL,
FFmpeg, or a renderer, and it refuses incomplete or internally mismatched
bundles before replacing the sidecar.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np

from inpainting.contracts import VideoGeometry, validate_depth_file, validate_mask_file

from .assets import resolve_robot_assets, validate_finger_trajectories
from .container_runner import DEFAULT_IMAGE, resolve_local_image_id
from .inputs import load_render_inputs
from .provenance import (
    PROVENANCE_SCHEMA,
    build_provenance,
    sha256_file,
    validate_sha256,
    verify_file_record,
)


RENDER_METADATA_SCHEMA = "v2d.inpainting.robot-render/v1"
ENRICHMENT_SCHEMA = "v2d.inpainting.robot-render-metadata-enrichment/v1"
ARTIFACT_NAMES = {
    "rgb": "robot_rgb.mp4",
    "mask": "robot_mask.npy",
    "depth": "robot_depth.npy",
}
_ASSET_SUMMARY_FIELDS = (
    "link_count",
    "actuated_joint_count",
    "actuated_joint_names",
    "mesh_count",
    "mesh_bytes",
)


class MetadataEnrichmentError(ValueError):
    """Raised before any write when a claimed bundle cannot be verified."""


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MetadataEnrichmentError(f"{label} must be a positive integer")
    return value


def _geometry(value: object) -> VideoGeometry:
    if not isinstance(value, dict):
        raise MetadataEnrichmentError("metadata geometry must be an object")
    frame_count = _positive_int(value.get("frame_count"), label="geometry.frame_count")
    width = _positive_int(value.get("width"), label="geometry.width")
    height = _positive_int(value.get("height"), label="geometry.height")
    fps = value.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise MetadataEnrichmentError("geometry.fps must be a positive finite number")
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise MetadataEnrichmentError("geometry.fps must be a positive finite number")
    return VideoGeometry(frame_count=frame_count, width=width, height=height, fps=fps)


def _verify_declared_input(
    metadata: dict[str, Any], key: str, source: Path, mounted_name: str
) -> str:
    declared = metadata.get(key)
    if not isinstance(declared, str):
        raise MetadataEnrichmentError(f"metadata {key} path must be a string")
    allowed = {str(source.resolve()), f"/inputs/{mounted_name}"}
    if declared not in allowed:
        raise MetadataEnrichmentError(
            f"metadata {key} path {declared!r} does not identify supplied file "
            f"{source.resolve()} or its renderer mount /inputs/{mounted_name}"
        )
    return declared


def _verify_video(path: Path, geometry: VideoGeometry) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise MetadataEnrichmentError(f"cannot open completed RGB video: {path}")
    decoded = 0
    try:
        reported_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        reported_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (geometry.height, geometry.width, 3):
                raise MetadataEnrichmentError(
                    f"RGB frame {decoded} has shape {frame.shape}, expected "
                    f"{(geometry.height, geometry.width, 3)}"
                )
            decoded += 1
    finally:
        capture.release()
    if (reported_width, reported_height, decoded) != (
        geometry.width,
        geometry.height,
        geometry.frame_count,
    ):
        raise MetadataEnrichmentError(
            "decoded RGB geometry/count does not match metadata: "
            f"{reported_width}x{reported_height}/{decoded} vs "
            f"{geometry.width}x{geometry.height}/{geometry.frame_count}"
        )
    if not np.isclose(
        reported_fps,
        geometry.fps,
        atol=max(1e-3, geometry.fps * 1e-4),
        rtol=0.0,
    ):
        raise MetadataEnrichmentError(
            f"decoded RGB fps {reported_fps:.8g} != metadata {geometry.fps:.8g}"
        )


def _verify_recorded_video_statistics(
    metadata: dict[str, Any], geometry: VideoGeometry
) -> None:
    statistics = metadata.get("render_statistics")
    if not isinstance(statistics, dict):
        raise MetadataEnrichmentError("metadata lacks render_statistics")
    video = statistics.get("video_verification")
    if not isinstance(video, dict):
        raise MetadataEnrichmentError("metadata lacks render_statistics.video_verification")
    for key, value in {
        "decoded_frame_count": geometry.frame_count,
        "width": geometry.width,
        "height": geometry.height,
    }.items():
        if video.get(key) != value:
            raise MetadataEnrichmentError(
                f"recorded video_verification.{key}={video.get(key)!r} != {value}"
            )
    fps = video.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isclose(
        float(fps), geometry.fps, atol=max(1e-3, geometry.fps * 1e-4), rtol=0.0
    ):
        raise MetadataEnrichmentError("recorded video_verification.fps mismatches geometry")


def _verify_mask_statistics(metadata: dict[str, Any], mask: np.ndarray) -> None:
    statistics = metadata["render_statistics"]
    counts = np.asarray(
        [np.count_nonzero(mask[index]) for index in range(mask.shape[0])],
        dtype=np.int64,
    )
    for key, value in {
        "robot_pixel_count": int(counts.sum()),
        "min_robot_pixels_per_frame": int(counts.min()),
        "max_robot_pixels_per_frame": int(counts.max()),
    }.items():
        if statistics.get(key) != value:
            raise MetadataEnrichmentError(
                f"recorded {key}={statistics.get(key)!r} != mask-derived {value}"
            )
    mean = statistics.get("mean_robot_pixels_per_frame")
    if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not np.isclose(
        float(mean), float(counts.mean()), atol=1e-9, rtol=0.0
    ):
        raise MetadataEnrichmentError("recorded mean robot pixels does not match mask")
    threshold = statistics.get("visibility_pixel_threshold")
    required = statistics.get("required_visible_frame_count")
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
        raise MetadataEnrichmentError("recorded visibility threshold must be positive")
    if isinstance(required, bool) or not isinstance(required, int) or required <= 0:
        raise MetadataEnrichmentError("recorded required visible frames must be positive")
    visible = int(np.count_nonzero(counts >= threshold))
    if statistics.get("visible_frame_count") != visible:
        raise MetadataEnrichmentError(
            "recorded visible frame count does not match the completed robot mask"
        )
    if visible < required:
        raise MetadataEnrichmentError(
            f"bundle records only {visible} visible frames but requires {required}"
        )


def _verify_assets(
    metadata: dict[str, Any], asset_root: Path
) -> tuple[dict[str, Any], Any]:
    assets = resolve_robot_assets(asset_root)
    actual = assets.as_dict()
    recorded = metadata.get("assets")
    if not isinstance(recorded, dict):
        raise MetadataEnrichmentError("metadata assets must be an object")
    if recorded.get("root") not in {str(asset_root.resolve()), "/external_assets"}:
        raise MetadataEnrichmentError(
            f"metadata asset root {recorded.get('root')!r} does not match supplied tree"
        )
    for part in ("arms", "left_hand", "right_hand"):
        old_part = recorded.get(part)
        new_part = actual[part]
        if not isinstance(old_part, dict):
            raise MetadataEnrichmentError(f"metadata assets.{part} must be an object")
        allowed_paths = {
            str(getattr(assets, part).path),
            f"/external_assets/{new_part['urdf_file']['path']}",
        }
        if old_part.get("path") not in allowed_paths:
            raise MetadataEnrichmentError(
                f"metadata assets.{part}.path does not match the supplied asset tree"
            )
        for field in _ASSET_SUMMARY_FIELDS:
            if old_part.get(field) != new_part[field]:
                raise MetadataEnrichmentError(
                    f"metadata assets.{part}.{field} differs from supplied asset tree"
                )
        if "urdf_file" in old_part and old_part["urdf_file"] != new_part["urdf_file"]:
            raise MetadataEnrichmentError(
                f"existing assets.{part}.urdf_file provenance is stale or mismatched"
            )
        if (
            "referenced_asset_files" in old_part
            and old_part["referenced_asset_files"] != new_part["referenced_asset_files"]
        ):
            raise MetadataEnrichmentError(
                f"existing assets.{part}.referenced_asset_files provenance is stale or mismatched"
            )
    return actual, assets


def _verify_existing_provenance(
    existing: object,
    actual: dict[str, Any],
    *,
    trajectory: Path,
    intrinsic: Path,
    world_to_camera: Path,
) -> None:
    if not isinstance(existing, dict):
        raise MetadataEnrichmentError("existing provenance must be an object")
    if existing.get("schema_version") != PROVENANCE_SCHEMA:
        raise MetadataEnrichmentError("existing provenance schema is unsupported")
    if existing.get("hash_algorithm") != "sha256":
        raise MetadataEnrichmentError("existing provenance hash algorithm is not sha256")
    capture_mode = existing.get("capture_mode")
    if capture_mode not in {"render_time", "retrospective_enrichment"}:
        raise MetadataEnrichmentError(
            f"existing provenance capture_mode is invalid: {capture_mode!r}"
        )
    if actual.get("capture_mode") != capture_mode:
        raise MetadataEnrichmentError("existing provenance capture mode changed unexpectedly")
    inputs = existing.get("inputs")
    if not isinstance(inputs, dict):
        raise MetadataEnrichmentError("existing provenance inputs must be an object")
    for key, path in (
        ("trajectory", trajectory),
        ("intrinsic", intrinsic),
        ("world_to_camera", world_to_camera),
    ):
        verify_file_record(path, inputs.get(key), label=f"input {key}")
    if existing.get("renderer_source_files") != actual["renderer_source_files"]:
        raise MetadataEnrichmentError(
            "existing renderer source provenance differs from the requested repository"
        )


def _atomic_json_replace(path: Path, payload: dict[str, Any], *, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode & 0o777)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def enrich_render_metadata(
    *,
    metadata_path: str | Path,
    trajectory: str | Path,
    intrinsic: str | Path,
    world_to_camera: str | Path,
    asset_root: str | Path,
    repository_root: str | Path,
    image: str,
    image_id: str,
    write: bool = True,
) -> dict[str, Any]:
    """Verify one bundle and optionally replace only its metadata sidecar."""

    metadata_source = Path(metadata_path).resolve()
    trajectory_source = Path(trajectory).resolve()
    intrinsic_source = Path(intrinsic).resolve()
    w2c_source = Path(world_to_camera).resolve()
    asset_source = Path(asset_root).resolve()
    repository_source = Path(repository_root).resolve()
    if not metadata_source.is_file():
        raise FileNotFoundError(metadata_source)
    original_bytes = metadata_source.read_bytes()
    original_stat = metadata_source.stat()
    try:
        metadata = json.loads(original_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataEnrichmentError(f"invalid metadata JSON: {metadata_source}") from exc
    if not isinstance(metadata, dict):
        raise MetadataEnrichmentError("render metadata must be a JSON object")
    if metadata.get("schema_version") != RENDER_METADATA_SCHEMA:
        raise MetadataEnrichmentError(
            f"metadata schema must be {RENDER_METADATA_SCHEMA!r}"
        )
    if metadata.get("state") != "complete":
        raise MetadataEnrichmentError("refusing metadata enrichment for non-complete bundle")
    bundle_root = metadata_source.parent
    host_output = metadata.get("host_output_dir")
    if not isinstance(host_output, str) or Path(host_output).resolve() != bundle_root:
        raise MetadataEnrichmentError(
            "metadata host_output_dir must resolve to the sidecar directory"
        )
    if metadata.get("container_image") != image:
        raise MetadataEnrichmentError(
            f"metadata image {metadata.get('container_image')!r} != requested {image!r}"
        )
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise MetadataEnrichmentError("image_id must be an immutable sha256: Docker ID")
    validate_sha256(image_id.removeprefix("sha256:"), label="container image ID")
    existing_image_id = metadata.get("container_image_id")
    if existing_image_id is not None and existing_image_id != image_id:
        raise MetadataEnrichmentError(
            f"existing container image ID {existing_image_id!r} != resolved {image_id!r}"
        )

    geometry = _geometry(metadata.get("geometry"))
    declared_paths = {
        "trajectory": _verify_declared_input(
            metadata, "trajectory", trajectory_source, "robot_trajectory.npz"
        ),
        "intrinsic": _verify_declared_input(
            metadata, "intrinsic", intrinsic_source, f"intrinsic{intrinsic_source.suffix}"
        ),
        "world_to_camera": _verify_declared_input(
            metadata,
            "world_to_camera",
            w2c_source,
            f"world_to_camera{w2c_source.suffix}",
        ),
    }
    inputs = load_render_inputs(
        trajectory_path=trajectory_source,
        intrinsic_path=intrinsic_source,
        world_to_camera_path=w2c_source,
        width=geometry.width,
        height=geometry.height,
        fps=geometry.fps,
    )
    if metadata.get("trajectory_coordinate_frame") != inputs.coordinate_frame:
        raise MetadataEnrichmentError(
            "metadata trajectory coordinate frame differs from supplied trajectory"
        )
    if metadata.get("projection_validation") != inputs.projection_report():
        raise MetadataEnrichmentError(
            "metadata wrist projection report differs from supplied inputs/calibration"
        )
    assets_dict, assets = _verify_assets(metadata, asset_source)
    validate_finger_trajectories(assets, inputs.trajectory)

    artifacts_value = metadata.get("artifacts")
    sizes_value = metadata.get("artifact_bytes")
    if not isinstance(artifacts_value, dict) or not isinstance(sizes_value, dict):
        raise MetadataEnrichmentError("metadata must declare artifacts and artifact_bytes")
    artifact_paths: dict[str, Path] = {}
    artifact_stats: dict[str, tuple[int, int, int, int]] = {}
    for key, name in ARTIFACT_NAMES.items():
        declared = artifacts_value.get(key)
        if not isinstance(declared, str) or Path(declared).name != name:
            raise MetadataEnrichmentError(
                f"metadata artifact {key} must declare basename {name!r}"
            )
        path = (bundle_root / name).resolve()
        if path.parent != bundle_root or not path.is_file() or path.stat().st_size <= 0:
            raise MetadataEnrichmentError(f"completed artifact is missing or empty: {path}")
        recorded_size = sizes_value.get(key)
        if (
            isinstance(recorded_size, bool)
            or not isinstance(recorded_size, int)
            or recorded_size != path.stat().st_size
        ):
            raise MetadataEnrichmentError(f"artifact byte count mismatch for {key}")
        stat = path.stat()
        artifact_paths[key] = path
        artifact_stats[key] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if original_stat.st_mtime_ns < max(
        path.stat().st_mtime_ns for path in artifact_paths.values()
    ):
        raise MetadataEnrichmentError(
            "render metadata predates one or more artifacts; bundle is not a safe commit"
        )

    _verify_recorded_video_statistics(metadata, geometry)
    _verify_video(artifact_paths["rgb"], geometry)
    mask = validate_mask_file(artifact_paths["mask"], geometry)
    validate_depth_file(artifact_paths["depth"], mask, geometry, name="Robot depth")
    _verify_mask_statistics(metadata, mask)

    artifact_hashes = {key: sha256_file(path) for key, path in artifact_paths.items()}
    existing_hashes = metadata.get("artifact_sha256")
    if existing_hashes is not None:
        if not isinstance(existing_hashes, dict):
            raise MetadataEnrichmentError("existing artifact_sha256 must be an object")
        for key, actual_hash in artifact_hashes.items():
            expected = validate_sha256(
                existing_hashes.get(key), label=f"artifact_sha256.{key}"
            )
            if expected != actual_hash:
                raise MetadataEnrichmentError(f"existing artifact SHA-256 mismatch for {key}")

    provenance = build_provenance(
        trajectory=trajectory_source,
        intrinsic=intrinsic_source,
        world_to_camera=w2c_source,
        recorded_paths=declared_paths,
        capture_mode=(
            str(metadata["provenance"].get("capture_mode"))
            if isinstance(metadata.get("provenance"), dict)
            else "retrospective_enrichment"
        ),
        package_root=repository_source / "inpainting" / "robot_renderer",
    )
    if "provenance" in metadata:
        _verify_existing_provenance(
            metadata["provenance"],
            provenance,
            trajectory=trajectory_source,
            intrinsic=intrinsic_source,
            world_to_camera=w2c_source,
        )

    # Recheck every provenance source plus artifact identity after the expensive
    # scans so concurrent mutation cannot be blessed by the sidecar.
    for key, path in (
        ("trajectory", trajectory_source),
        ("intrinsic", intrinsic_source),
        ("world_to_camera", w2c_source),
    ):
        verify_file_record(path, provenance["inputs"][key], label=f"input {key}")
    for record in provenance["renderer_source_files"]:
        verify_file_record(
            repository_source / record["path"],
            record,
            label=f"renderer source {record['path']}",
        )
    for part in ("arms", "left_hand", "right_hand"):
        verify_file_record(
            getattr(assets, part).path,
            assets_dict[part]["urdf_file"],
            label=f"{part} URDF",
        )
        for record in assets_dict[part]["referenced_asset_files"]:
            verify_file_record(
                asset_source / record["path"],
                record,
                label=f"{part} asset {record['path']}",
            )
    for key, path in artifact_paths.items():
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if signature != artifact_stats[key] or sha256_file(path) != artifact_hashes[key]:
            raise MetadataEnrichmentError(f"artifact changed during verification: {key}")
    if metadata_source.read_bytes() != original_bytes:
        raise MetadataEnrichmentError("metadata changed concurrently during verification")

    enriched = dict(metadata)
    enriched["container_image_id"] = image_id
    enriched["provenance"] = provenance
    recorded_assets = metadata["assets"]
    enriched_assets = dict(recorded_assets)
    for part in ("arms", "left_hand", "right_hand"):
        enriched_part = dict(recorded_assets[part])
        enriched_part["urdf_file"] = assets_dict[part]["urdf_file"]
        enriched_part["referenced_asset_files"] = assets_dict[part][
            "referenced_asset_files"
        ]
        enriched_assets[part] = enriched_part
    enriched["assets"] = enriched_assets
    enriched["artifact_sha256"] = artifact_hashes
    enriched["provenance_enrichment"] = {
        "schema_version": ENRICHMENT_SCHEMA,
        "operation": "verified_metadata_only_no_rerender",
        "atomic_replacement": True,
        "capture_mode": provenance["capture_mode"],
    }
    if write:
        _atomic_json_replace(metadata_source, enriched, mode=original_stat.st_mode)
    return enriched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--world-to-camera", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="perform every check and hash but leave the sidecar unchanged",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_id = resolve_local_image_id(args.image)
    enriched = enrich_render_metadata(
        metadata_path=args.metadata,
        trajectory=args.trajectory,
        intrinsic=args.intrinsics,
        world_to_camera=args.world_to_camera,
        asset_root=args.asset_root,
        repository_root=args.repository_root,
        image=args.image,
        image_id=image_id,
        write=not args.verify_only,
    )
    print(
        json.dumps(
            {
                "state": "verified" if args.verify_only else "enriched",
                "metadata": str(args.metadata.resolve()),
                "container_image": enriched["container_image"],
                "container_image_id": enriched["container_image_id"],
                "capture_mode": enriched["provenance"]["capture_mode"],
                "artifact_sha256": enriched["artifact_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
