"""Safely composite a validated robot-render bundle over an inpainted video."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

import cv2
import numpy as np

from .contracts import (
    ContractError,
    TACO_OBJECT_RENDER_SCHEMA,
    VideoGeometry,
    validate_depth_file,
    validate_mask_file,
)
from .occluder_depth import (
    OCCLUDER_DEPTH_SCHEMA,
    validate_occluder_depth_bundle,
)
from .robot_renderer.backend import RENDER_METADATA_SCHEMA
from .video_io import probe_video


COMPOSITE_SCHEMA = "v2d.inpainting.composite/v1"
OBJECT_RENDER_SCHEMA = TACO_OBJECT_RENDER_SCHEMA
ROBOT_ARTIFACT_NAMES = {
    "rgb": "robot_rgb.mp4",
    "mask": "robot_mask.npy",
    "depth": "robot_depth.npy",
}
OBJECT_ARTIFACT_NAMES = {
    "mask": "object_mask.npy",
    "depth": "object_depth.npy",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    status = path.stat()
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def _fingerprint(path: Path) -> dict[str, int | str]:
    before = _stat_signature(path)
    digest = _sha256(path)
    if _stat_signature(path) != before:
        raise ContractError(f"Input changed while being fingerprinted: {path}")
    return {"bytes": before[2], "sha256": digest}


def _validated_bundle_fingerprint(
    metadata: dict[str, Any], key: str, path: Path
) -> dict[str, int | str]:
    """Reuse a hash already checked by `_load_bundle`, otherwise compute it."""

    hashes = metadata.get("artifact_sha256")
    sizes = metadata.get("artifact_bytes")
    if isinstance(hashes, dict) and isinstance(hashes.get(key), str):
        return {"bytes": int(sizes[key]), "sha256": hashes[key]}
    return _fingerprint(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _geometry_from_metadata(value: Any, *, label: str) -> VideoGeometry:
    if not isinstance(value, dict):
        raise ContractError(f"{label} geometry must be an object")
    required = ("frame_count", "width", "height", "fps")
    if any(key not in value for key in required):
        raise ContractError(f"{label} geometry is missing required keys")
    integers = []
    for key in required[:3]:
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ContractError(f"{label} geometry {key} must be a positive integer")
        integers.append(item)
    fps = value["fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise ContractError(f"{label} geometry fps must be numeric")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ContractError(f"{label} geometry fps must be positive and finite")
    return VideoGeometry(
        frame_count=integers[0], width=integers[1], height=integers[2], fps=fps
    )


def _same_geometry(left: VideoGeometry, right: VideoGeometry) -> bool:
    return (
        left.frame_count == right.frame_count
        and left.width == right.width
        and left.height == right.height
        and abs(left.fps - right.fps) <= 1e-3
    )


def _load_bundle(
    metadata_path: Path,
    *,
    expected_schema: str,
    expected_artifacts: dict[str, str],
    geometry: VideoGeometry,
    label: str,
    require_hashes: bool,
) -> tuple[dict[str, Any], dict[str, Path]]:
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Could not read {label} bundle metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ContractError(f"{label} bundle metadata must be a JSON object")
    if metadata.get("schema_version") != expected_schema:
        raise ContractError(
            f"{label} bundle schema must be {expected_schema!r}, got "
            f"{metadata.get('schema_version')!r}"
        )
    if metadata.get("state") != "complete":
        raise ContractError(f"{label} bundle state must be 'complete'")
    bundle_geometry = _geometry_from_metadata(metadata.get("geometry"), label=label)
    if not _same_geometry(bundle_geometry, geometry):
        raise ContractError(
            f"{label} bundle geometry {bundle_geometry} does not match source {geometry}"
        )

    bundle_root = metadata_path.parent.resolve()
    host_output = metadata.get("host_output_dir")
    if host_output is not None:
        if not isinstance(host_output, str) or Path(host_output).resolve() != bundle_root:
            raise ContractError(
                f"{label} host_output_dir does not resolve to metadata directory"
            )
    artifacts_value = metadata.get("artifacts")
    sizes_value = metadata.get("artifact_bytes")
    hashes_value = metadata.get("artifact_sha256")
    if not isinstance(artifacts_value, dict) or not isinstance(sizes_value, dict):
        raise ContractError(f"{label} bundle must declare artifacts and artifact_bytes")
    if require_hashes and not isinstance(hashes_value, dict):
        raise ContractError(f"{label} bundle must declare artifact_sha256")

    resolved: dict[str, Path] = {}
    for key, expected_name in expected_artifacts.items():
        declared = artifacts_value.get(key)
        if (
            not isinstance(declared, str)
            or ".." in Path(declared).parts
            or Path(declared).name != expected_name
        ):
            raise ContractError(
                f"{label} artifact {key!r} must declare basename {expected_name!r}"
            )
        path = (bundle_root / expected_name).resolve()
        try:
            path.relative_to(bundle_root)
        except ValueError as exc:
            raise ContractError(f"{label} artifact escapes its bundle directory") from exc
        if not path.is_file():
            raise FileNotFoundError(f"{label} artifact: {path}")
        expected_size = sizes_value.get(key)
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
        ):
            raise ContractError(
                f"{label} artifact_bytes[{key!r}] must be a positive integer"
            )
        actual_fingerprint = _fingerprint(path)
        if expected_size != actual_fingerprint["bytes"]:
            raise ContractError(
                f"{label} artifact size mismatch for {key}: metadata {expected_size}, "
                f"file {actual_fingerprint['bytes']}"
            )
        if isinstance(hashes_value, dict):
            expected_hash = hashes_value.get(key)
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(character not in "0123456789abcdef" for character in expected_hash)
            ):
                raise ContractError(f"{label} artifact_sha256[{key!r}] is invalid")
            if actual_fingerprint["sha256"] != expected_hash:
                raise ContractError(f"{label} artifact SHA-256 mismatch for {key}")
        resolved[key] = path
    return metadata, resolved


def validate_robot_render_bundle(
    metadata_path: str | Path, geometry: VideoGeometry
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Validate a committed robot-render bundle and return its exact artifacts."""

    return _load_bundle(
        Path(metadata_path).resolve(),
        expected_schema=RENDER_METADATA_SCHEMA,
        expected_artifacts=ROBOT_ARTIFACT_NAMES,
        geometry=geometry,
        label="Robot render",
        require_hashes=True,
    )


def validate_taco_object_render_bundle(
    metadata_path: str | Path, geometry: VideoGeometry
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Validate a committed TACO object-render bundle including SHA-256s."""

    return _load_bundle(
        Path(metadata_path).resolve(),
        expected_schema=OBJECT_RENDER_SCHEMA,
        expected_artifacts=OBJECT_ARTIFACT_NAMES,
        geometry=geometry,
        label="TACO object render",
        require_hashes=True,
    )


def validate_object_occluder_bundle(
    metadata_path: str | Path, geometry: VideoGeometry
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Dispatch a depth occluder without mislabelling estimates as TACO GT."""

    path = Path(metadata_path).resolve()
    try:
        candidate = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Could not read object occluder metadata: {path}") from exc
    if not isinstance(candidate, dict):
        raise ContractError("Object occluder metadata must be a JSON object")
    schema = candidate.get("schema_version")
    if schema == OBJECT_RENDER_SCHEMA:
        return validate_taco_object_render_bundle(path, geometry)
    if schema == OCCLUDER_DEPTH_SCHEMA:
        return validate_occluder_depth_bundle(path, geometry)
    raise ContractError(
        "Object occluder schema must be one of "
        f"{(OBJECT_RENDER_SCHEMA, OCCLUDER_DEPTH_SCHEMA)!r}, got {schema!r}"
    )


def _assert_distinct_paths(paths: dict[str, Path]) -> None:
    seen: dict[Path, str] = {}
    for label, path in paths.items():
        resolved = path.resolve()
        if resolved in seen:
            raise ContractError(
                f"Path alias is not allowed: {label} and {seen[resolved]} both resolve "
                f"to {resolved}"
            )
        seen[resolved] = label


def _assert_stat_signatures_unchanged(
    paths: dict[str, Path], signatures: dict[str, tuple[int, int, int, int]]
) -> None:
    for label, path in paths.items():
        if _stat_signature(path) != signatures[label]:
            raise ContractError(f"{label} changed while compositing: {path}")


def _assert_bundle_unchanged(
    *,
    metadata_path: Path,
    initial_metadata_fingerprint: dict[str, int | str],
    initial_metadata: dict[str, Any],
    initial_artifacts: dict[str, Path],
    geometry: VideoGeometry,
    bundle_type: str,
) -> None:
    """Rehash a consumed bundle and reject any generation change before commit."""

    if _fingerprint(metadata_path) != initial_metadata_fingerprint:
        raise ContractError(f"Input metadata changed while compositing: {metadata_path}")
    if bundle_type == "robot":
        metadata, artifacts = validate_robot_render_bundle(metadata_path, geometry)
    elif bundle_type == "occluder":
        metadata, artifacts = validate_object_occluder_bundle(metadata_path, geometry)
    else:
        raise ValueError(f"Unsupported bundle_type: {bundle_type!r}")
    if metadata != initial_metadata or artifacts != initial_artifacts:
        raise ContractError(f"Input bundle changed while compositing: {metadata_path}")
    # Bracket artifact verification so a sidecar replacement during hashing is
    # detected even when both generations are independently valid.
    if _fingerprint(metadata_path) != initial_metadata_fingerprint:
        raise ContractError(f"Input metadata changed while compositing: {metadata_path}")


def depth_visible_robot_mask(
    robot_mask: np.ndarray,
    robot_depth: np.ndarray,
    object_mask: np.ndarray,
    object_depth: np.ndarray,
    *,
    depth_guard_m: float = 0.003,
) -> np.ndarray:
    """Return robot pixels not hidden by a closer metric-depth occluder."""

    if (
        isinstance(depth_guard_m, bool)
        or not np.isfinite(depth_guard_m)
        or depth_guard_m < 0
    ):
        raise ValueError("depth_guard_m must be a finite non-negative number")
    robot_mask = np.asarray(robot_mask)
    object_mask = np.asarray(object_mask)
    robot_depth = np.asarray(robot_depth)
    object_depth = np.asarray(object_depth)
    if robot_mask.dtype != np.bool_ or object_mask.dtype != np.bool_:
        raise ContractError("Depth compositing masks must have boolean dtype")
    if not (
        robot_mask.shape
        == object_mask.shape
        == robot_depth.shape
        == object_depth.shape
    ):
        raise ContractError("Depth compositing arrays must have identical shapes")
    return robot_mask & (
        ~object_mask | (robot_depth <= object_depth + float(depth_guard_m))
    )


def composite_robot(
    base_video: Path,
    robot_video: Path,
    robot_mask_path: Path,
    output_video: Path,
    *,
    robot_metadata_path: Path,
    metadata_path: Path | None = None,
    object_mask_path: Path | None = None,
    object_depth_path: Path | None = None,
    object_metadata_path: Path | None = None,
    depth_guard_m: float = 0.003,
    overwrite: bool = False,
) -> dict[str, Any]:
    base_video = Path(base_video).resolve()
    robot_video = Path(robot_video).resolve()
    robot_mask_path = Path(robot_mask_path).resolve()
    robot_metadata_path = Path(robot_metadata_path).resolve()
    output_video = Path(output_video).resolve()
    if metadata_path is None:
        metadata_path = output_video.with_suffix(".json")
    metadata_path = Path(metadata_path).resolve()
    if output_video.suffix.lower() != ".mp4":
        raise ContractError("Composite output must use an .mp4 suffix")
    for path in (base_video, robot_video, robot_mask_path, robot_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    depth_values = (object_mask_path, object_depth_path, object_metadata_path)
    if any(value is not None for value in depth_values) and not all(
        value is not None for value in depth_values
    ):
        raise ContractError(
            "object_mask_path, object_depth_path, and object_metadata_path are all-or-none"
        )
    depth_aware = all(value is not None for value in depth_values)
    if isinstance(depth_guard_m, bool) or not np.isfinite(depth_guard_m) or depth_guard_m < 0:
        raise ValueError("depth_guard_m must be a finite non-negative number")
    object_paths: tuple[Path, Path, Path] | None = None
    if depth_aware:
        assert object_mask_path is not None
        assert object_depth_path is not None
        assert object_metadata_path is not None
        object_paths = (
            Path(object_mask_path).resolve(),
            Path(object_depth_path).resolve(),
            Path(object_metadata_path).resolve(),
        )
        for path in object_paths:
            if not path.is_file():
                raise FileNotFoundError(path)

    alias_candidates = {
        "base_video": base_video,
        "robot_video": robot_video,
        "robot_mask": robot_mask_path,
        "robot_metadata": robot_metadata_path,
        "output_video": output_video,
        "output_metadata": metadata_path,
    }
    if object_paths is not None:
        alias_candidates.update(
            {
                "object_mask": object_paths[0],
                "object_depth": object_paths[1],
                "object_metadata": object_paths[2],
            }
        )
    _assert_distinct_paths(alias_candidates)
    existing_outputs = [path for path in (output_video, metadata_path) if path.exists()]
    if existing_outputs and not overwrite:
        raise FileExistsError(
            "composite outputs already exist; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    base_video_fingerprint = _fingerprint(base_video)
    base_geometry = probe_video(base_video)
    robot_geometry = probe_video(robot_video)
    if not _same_geometry(base_geometry, robot_geometry):
        raise ContractError(
            f"Base geometry {base_geometry} does not match robot geometry {robot_geometry}"
        )
    robot_metadata, robot_bundle = validate_robot_render_bundle(
        robot_metadata_path, base_geometry
    )
    robot_metadata_fingerprint = _fingerprint(robot_metadata_path)
    if robot_video != robot_bundle["rgb"] or robot_mask_path != robot_bundle["mask"]:
        raise ContractError(
            "Supplied robot video/mask do not resolve to the declared renderer bundle"
        )
    _assert_distinct_paths({**alias_candidates, "robot_depth": robot_bundle["depth"]})
    robot_masks = validate_mask_file(robot_mask_path, base_geometry)

    object_metadata: dict[str, Any] | None = None
    object_metadata_fingerprint: dict[str, int | str] | None = None
    object_bundle: dict[str, Path] | None = None
    object_masks = None
    object_depth = None
    robot_depth = None
    if object_paths is not None:
        object_mask_path, object_depth_path, object_metadata_path = object_paths
        object_metadata, object_bundle = validate_object_occluder_bundle(
            object_metadata_path, base_geometry
        )
        object_metadata_fingerprint = _fingerprint(object_metadata_path)
        if (
            object_mask_path != object_bundle["mask"]
            or object_depth_path != object_bundle["depth"]
        ):
            raise ContractError(
                "Supplied object mask/depth do not resolve to the declared object bundle"
            )
        object_masks = validate_mask_file(object_mask_path, base_geometry)
        object_depth = validate_depth_file(
            object_depth_path, object_masks, base_geometry, name="Object depth"
        )
        robot_depth = validate_depth_file(
            robot_bundle["depth"], robot_masks, base_geometry, name="Robot depth"
        )

    consumed_input_paths = {
        "base video": base_video,
        "robot video": robot_bundle["rgb"],
        "robot mask": robot_bundle["mask"],
        "robot depth": robot_bundle["depth"],
        "robot metadata": robot_metadata_path,
    }
    if object_paths is not None:
        assert object_bundle is not None
        consumed_input_paths.update(
            {
                "object mask": object_bundle["mask"],
                "object depth": object_bundle["depth"],
                "object metadata": object_paths[2],
            }
        )
    initial_stat_signatures = {
        label: _stat_signature(path) for label, path in consumed_input_paths.items()
    }

    output_video.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    temporary_video = output_video.with_name(
        f".{output_video.stem}.{run_id}.partial{output_video.suffix}"
    )
    if overwrite and metadata_path.exists():
        _write_json_atomic(
            metadata_path,
            {
                "schema_version": COMPOSITE_SCHEMA,
                "state": "committing",
                "run_id": run_id,
                "started_at": _utc_now(),
                "output_video": str(output_video),
            },
        )

    base_capture = cv2.VideoCapture(str(base_video))
    robot_capture = cv2.VideoCapture(str(robot_video))
    if not base_capture.isOpened() or not robot_capture.isOpened():
        base_capture.release()
        robot_capture.release()
        raise ContractError("Could not open a composite input video")
    writer = cv2.VideoWriter(
        str(temporary_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        base_geometry.fps,
        (base_geometry.width, base_geometry.height),
    )
    if not writer.isOpened():
        base_capture.release()
        robot_capture.release()
        temporary_video.unlink(missing_ok=True)
        raise RuntimeError(f"Could not open video writer: {temporary_video}")
    written = 0
    robot_pixels = 0
    object_occluded_pixels = 0
    try:
        for frame_index in range(base_geometry.frame_count):
            base_ok, base_frame = base_capture.read()
            robot_ok, robot_frame = robot_capture.read()
            if not base_ok or not robot_ok:
                raise ContractError(
                    f"Video decode ended at frame {frame_index}; expected "
                    f"{base_geometry.frame_count}"
                )
            mask = robot_masks[frame_index]
            if depth_aware:
                assert object_masks is not None
                assert object_depth is not None
                assert robot_depth is not None
                visible = depth_visible_robot_mask(
                    mask,
                    robot_depth[frame_index],
                    object_masks[frame_index],
                    object_depth[frame_index],
                    depth_guard_m=depth_guard_m,
                )
                object_occluded_pixels += int(mask.sum() - visible.sum())
                mask = visible
            robot_pixels += int(mask.sum())
            base_frame[mask] = robot_frame[mask]
            writer.write(base_frame)
            written += 1
        if base_capture.read()[0] or robot_capture.read()[0]:
            raise ContractError("A decoder returned extra frames beyond declared frame count")
    except Exception:
        temporary_video.unlink(missing_ok=True)
        raise
    finally:
        base_capture.release()
        robot_capture.release()
        writer.release()
    try:
        if written != base_geometry.frame_count:
            raise ContractError(f"Wrote {written} frames; expected {base_geometry.frame_count}")
        temporary_geometry = probe_video(temporary_video)
        if not _same_geometry(temporary_geometry, base_geometry):
            raise ContractError(
                f"Encoded composite geometry {temporary_geometry} does not match {base_geometry}"
            )

        if _fingerprint(base_video) != base_video_fingerprint:
            raise ContractError(f"Base video changed while compositing: {base_video}")
        _assert_bundle_unchanged(
            metadata_path=robot_metadata_path,
            initial_metadata_fingerprint=robot_metadata_fingerprint,
            initial_metadata=robot_metadata,
            initial_artifacts=robot_bundle,
            geometry=base_geometry,
            bundle_type="robot",
        )
        if object_paths is not None:
            assert object_metadata is not None
            assert object_metadata_fingerprint is not None
            assert object_bundle is not None
            _assert_bundle_unchanged(
                metadata_path=object_paths[2],
                initial_metadata_fingerprint=object_metadata_fingerprint,
                initial_metadata=object_metadata,
                initial_artifacts=object_bundle,
                geometry=base_geometry,
                bundle_type="occluder",
            )
        _assert_stat_signatures_unchanged(
            consumed_input_paths, initial_stat_signatures
        )

        input_fingerprints = {
            "base_video": base_video_fingerprint,
            "robot_metadata": robot_metadata_fingerprint,
            "robot_rgb": _validated_bundle_fingerprint(
                robot_metadata, "rgb", robot_bundle["rgb"]
            ),
            "robot_mask": _validated_bundle_fingerprint(
                robot_metadata, "mask", robot_bundle["mask"]
            ),
            "robot_depth": _validated_bundle_fingerprint(
                robot_metadata, "depth", robot_bundle["depth"]
            ),
        }
        if object_paths is not None:
            assert object_metadata is not None
            assert object_metadata_fingerprint is not None
            input_fingerprints.update(
                {
                    "object_mask": _validated_bundle_fingerprint(
                        object_metadata, "mask", object_paths[0]
                    ),
                    "object_depth": _validated_bundle_fingerprint(
                        object_metadata, "depth", object_paths[1]
                    ),
                    "object_metadata": object_metadata_fingerprint,
                }
            )

        temporary_video.replace(output_video)
        final_geometry = probe_video(output_video)
        if not _same_geometry(final_geometry, base_geometry):
            raise ContractError(
                f"Final composite geometry {final_geometry} does not match {base_geometry}"
            )
        metadata: dict[str, Any] = {
            "schema_version": COMPOSITE_SCHEMA,
            "state": "complete",
            "run_id": run_id,
            "completed_at": _utc_now(),
            "base_video": str(base_video),
            "robot_video": str(robot_video),
            "robot_mask": str(robot_mask_path),
            "robot_metadata": str(robot_metadata_path),
            "robot_render_run_id": robot_metadata.get("run_id"),
            "output_video": str(output_video),
            "geometry": base_geometry.as_dict(),
            "frames_written": written,
            "compositing": (
                "taco_object_depth"
                if object_metadata is not None
                and object_metadata.get("schema_version") == OBJECT_RENDER_SCHEMA
                else "estimated_occluder_depth"
                if depth_aware
                else "hard_robot_mask"
            ),
            "robot_pixels_written": robot_pixels,
            "object_occluded_robot_pixels": object_occluded_pixels,
            "input_fingerprints": input_fingerprints,
            "output_fingerprint": _fingerprint(output_video),
            "overwrite": bool(overwrite),
        }
        if object_paths is not None:
            metadata.update(
                {
                    "object_mask": str(object_paths[0]),
                    "object_depth": str(object_paths[1]),
                    "object_metadata": str(object_paths[2]),
                    "object_render_run_id": object_metadata.get("run_id")
                    if object_metadata is not None
                    else None,
                    "object_occluder_schema": object_metadata.get("schema_version")
                    if object_metadata is not None
                    else None,
                    "object_occluder_producer": object_metadata.get("producer")
                    if object_metadata is not None
                    else None,
                    "depth_guard_m": float(depth_guard_m),
                }
            )
        _write_json_atomic(metadata_path, metadata)
        return metadata
    finally:
        temporary_video.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-video", required=True, type=Path)
    parser.add_argument("--robot-video", required=True, type=Path)
    parser.add_argument("--robot-mask", required=True, type=Path)
    parser.add_argument("--robot-metadata", required=True, type=Path)
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--object-mask", type=Path)
    parser.add_argument("--object-depth", type=Path)
    parser.add_argument("--object-metadata", type=Path)
    parser.add_argument("--depth-guard-m", type=float, default=0.003)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = composite_robot(
        args.base_video,
        args.robot_video,
        args.robot_mask,
        args.output_video,
        robot_metadata_path=args.robot_metadata,
        metadata_path=args.metadata,
        object_mask_path=args.object_mask,
        object_depth_path=args.object_depth,
        object_metadata_path=args.object_metadata,
        depth_guard_m=args.depth_guard_m,
        overwrite=args.overwrite,
    )
    print(f"Wrote {metadata['frames_written']} frames -> {metadata['output_video']}")


if __name__ == "__main__":
    main()
