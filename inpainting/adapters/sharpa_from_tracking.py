"""Container-friendly Sharpa IK stage for MANO-enriched tracking archives."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from inpainting.adapters.video2data import (
    DEFAULT_ROBOT_ASSETS_DIR,
    AdapterError,
    ExistingSharpaBackend,
    _atomic_json,
    _module_available,
    _sha256,
    sharpa_asset_artifacts,
    sharpa_asset_blockers,
    trajectory_from_tracking,
)
from inpainting.adapters import video2data as video2data_module
from inpainting import contracts as contracts_module
from inpainting.contracts import (
    ContractError,
    validate_robot_trajectory_arrays,
    validate_tracking_arrays,
)


STAGE_SCHEMA = "v2d.inpainting.sharpa-from-tracking-stage/v2"
TRACKING_STAGE_SCHEMAS = {
    "v2d.inpainting.video2data-tracking-stage/v1",
    "v2d.inpainting.video2data-tracking-stage/v2",
}
PHANTOM_RUN_SCHEMA = "v2d.inpainting.phantom-run/v1"
_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def _validate_image_id(value: str) -> None:
    if not _IMAGE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "sharpa_image_id must be an immutable sha256:<64 lowercase hex> "
            "Docker image ID"
        )


def _file_artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _implementation_artifacts() -> dict[str, dict[str, Any]]:
    """Hash the local adapter plus the complete imported retarget package."""

    hand_kinematics = importlib.import_module(
        "robotic_grounding.retarget.hand_kinematics"
    )
    retarget_utils = importlib.import_module(
        "robotic_grounding.retarget.retarget_utils"
    )
    retarget_roots = {
        Path(str(hand_kinematics.__file__)).resolve().parent,
        Path(str(retarget_utils.__file__)).resolve().parent,
    }
    sources: dict[str, Path] = {
        "inpainting/adapters/sharpa_from_tracking.py": Path(__file__),
        "inpainting/adapters/video2data.py": Path(str(video2data_module.__file__)),
        "inpainting/contracts.py": Path(str(contracts_module.__file__)),
    }
    for root in sorted(retarget_roots, key=str):
        for path in sorted(root.rglob("*.py")):
            logical = f"robotic_grounding/retarget/{path.relative_to(root).as_posix()}"
            existing = sources.get(logical)
            if existing is not None and existing.resolve() != path.resolve():
                raise AdapterError(
                    f"Retarget source key collision for {logical}: {existing} and {path}"
                )
            sources[logical] = path
    return {key: _file_artifact(path) for key, path in sorted(sources.items())}


def _identity_view(artifacts: dict[str, dict[str, Any]]) -> dict[str, tuple[int, str]]:
    return {
        key: (int(value["size_bytes"]), str(value["sha256"]))
        for key, value in artifacts.items()
    }


def _assert_snapshot_unchanged(
    label: str,
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> None:
    if _identity_view(before) != _identity_view(after):
        raise AdapterError(
            f"{label} changed while Sharpa retargeting was running; "
            "refusing the atomic commit"
        )


def _load_tracking(path: Path) -> tuple[dict[str, np.ndarray], int]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise ContractError(f"Cannot read tracking archive {path}: {exc}") from exc
    frame_count = validate_tracking_arrays(arrays)
    for side in ("left", "right"):
        for key in (f"{side}_joints_3d", f"{side}_joints_wxyz"):
            if key not in arrays:
                raise ContractError(f"Sharpa stage requires {key}")
    return arrays, frame_count


def _load_tracking_metadata(path: Path, tracking_path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read tracking metadata {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ContractError("Tracking metadata must be a JSON object")
    schema = metadata.get("schema_version")
    if schema not in {*TRACKING_STAGE_SCHEMAS, PHANTOM_RUN_SCHEMA}:
        raise ContractError(f"Unsupported tracking metadata schema {schema!r}")
    if metadata.get("state") != "complete":
        raise ContractError("Tracking metadata is not a complete commit marker")
    if schema in TRACKING_STAGE_SCHEMAS:
        artifact = metadata.get("tracking")
        size_key = "size_bytes"
    else:
        outputs = metadata.get("outputs")
        artifact = outputs.get("tracking") if isinstance(outputs, dict) else None
        size_key = "bytes"
    if not isinstance(artifact, dict):
        raise ContractError("Tracking metadata omits tracking artifact identity")
    if schema == PHANTOM_RUN_SCHEMA and artifact.get("filename") != tracking_path.name:
        raise ContractError(
            "Phantom tracking filename does not match its commit marker"
        )
    if artifact.get(size_key) != tracking_path.stat().st_size:
        raise ContractError("Tracking byte size no longer matches tracking metadata")
    actual_hash = _sha256(tracking_path)
    if artifact.get("sha256") != actual_hash:
        raise ContractError("Tracking SHA-256 no longer matches tracking metadata")
    with np.load(tracking_path, allow_pickle=False) as archive:
        archive_tracker = str(np.asarray(archive["tracker"]).item())
        archive_frame = str(np.asarray(archive["coordinate_frame"]).item())
        archive_count = int(np.asarray(archive["frame_indices"]).shape[0])
    if metadata.get("tracker") != archive_tracker:
        raise ContractError("Tracking metadata tracker does not match the archive")
    if metadata.get("coordinate_frame") not in (None, archive_frame):
        raise ContractError(
            "Tracking metadata coordinate frame does not match the archive"
        )
    geometry = (
        metadata.get("video")
        if schema in TRACKING_STAGE_SCHEMAS
        else metadata.get("geometry")
    )
    if not isinstance(geometry, dict) or geometry.get("frame_count") != archive_count:
        raise ContractError("Tracking metadata geometry does not match the archive")
    return metadata


def preflight_sharpa(
    *,
    tracking: Path,
    tracking_metadata: Path,
    output_dir: Path,
    robot_assets_dir: Path,
    sharpa_image_id: str,
    overwrite: bool,
) -> dict[str, Any]:
    tracking = tracking.expanduser().resolve()
    tracking_metadata = tracking_metadata.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    robot_assets_dir = robot_assets_dir.expanduser().resolve()
    blockers: list[dict[str, str]] = []
    frame_count: int | None = None
    source_metadata: dict[str, Any] | None = None
    implementation_sources: dict[str, dict[str, Any]] = {}
    asset_artifacts: dict[str, dict[str, Any]] = {}
    try:
        _validate_image_id(sharpa_image_id)
    except ValueError as exc:
        blockers.append(
            {
                "code": "invalid_sharpa_image_id",
                "path": "--sharpa-image-id",
                "detail": str(exc),
            }
        )
    if not tracking.is_file():
        blockers.append(
            {
                "code": "missing_tracking",
                "path": str(tracking),
                "detail": "tracking.npz is required.",
            }
        )
    else:
        try:
            _, frame_count = _load_tracking(tracking)
        except ContractError as exc:
            blockers.append(
                {"code": "invalid_tracking", "path": str(tracking), "detail": str(exc)}
            )
    if not tracking_metadata.is_file():
        blockers.append(
            {
                "code": "missing_tracking_metadata",
                "path": str(tracking_metadata),
                "detail": "tracking.json complete commit marker is required.",
            }
        )
    elif tracking.is_file():
        try:
            source_metadata = _load_tracking_metadata(tracking_metadata, tracking)
        except ContractError as exc:
            blockers.append(
                {
                    "code": "invalid_tracking_metadata",
                    "path": str(tracking_metadata),
                    "detail": str(exc),
                }
            )
    required_modules = (
        "torch",
        "scipy",
        "pinocchio",
        "pink",
        "loop_rate_limiters",
        "qpsolvers",
        "daqp",
        "robotic_grounding.retarget.retarget_utils",
    )
    missing = [name for name in required_modules if not _module_available(name)]
    if missing:
        blockers.append(
            {
                "code": "missing_sharpa_dependencies",
                "path": "environment",
                "detail": "Missing: " + ", ".join(missing),
            }
        )
    else:
        try:
            implementation_sources = _implementation_artifacts()
        except (AdapterError, ImportError, OSError, ValueError) as exc:
            blockers.append(
                {
                    "code": "unverifiable_sharpa_implementation",
                    "path": "environment",
                    "detail": str(exc),
                }
            )
    asset_blockers = sharpa_asset_blockers(robot_assets_dir)
    blockers.extend(asset_blockers)
    if not asset_blockers:
        try:
            asset_artifacts = sharpa_asset_artifacts(robot_assets_dir)
        except (AdapterError, OSError, ValueError) as exc:
            blockers.append(
                {
                    "code": "unverifiable_sharpa_assets",
                    "path": str(robot_assets_dir),
                    "detail": str(exc),
                }
            )
    outputs = [
        output_dir / "robot_trajectory.npz",
        output_dir / "robot_trajectory.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        blockers.append(
            {
                "code": "outputs_exist",
                "path": str(output_dir),
                "detail": "Refusing to replace: "
                + ", ".join(path.name for path in existing),
            }
        )
    report: dict[str, Any] = {
        "schema_version": STAGE_SCHEMA,
        "mode": "preflight",
        "state": "ready" if not blockers else "blocked",
        "tracking": str(tracking),
        "tracking_metadata": str(tracking_metadata),
        "sharpa_image_id": sharpa_image_id,
        "robot_assets_dir": str(robot_assets_dir),
        "implementation_sources": implementation_sources,
        "robot_assets": asset_artifacts,
        "outputs": [str(path) for path in outputs],
        "blockers": blockers,
    }
    if frame_count is not None:
        report["frame_count"] = frame_count
    if source_metadata is not None:
        report["sequence_id"] = source_metadata.get("sequence_id")
    return report


def execute_sharpa(
    *,
    tracking: Path,
    tracking_metadata: Path,
    output_dir: Path,
    robot_assets_dir: Path,
    sharpa_image_id: str,
    device: str,
    mano_to_robot_scale: float,
    max_frame_task_error_m: float,
    overwrite: bool,
) -> dict[str, Any]:
    tracking = tracking.expanduser().resolve()
    tracking_metadata = tracking_metadata.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    robot_assets_dir = robot_assets_dir.expanduser().resolve()
    _validate_image_id(sharpa_image_id)
    tracking_snapshot = {
        "tracking": _file_artifact(tracking),
        "tracking_metadata": _file_artifact(tracking_metadata),
    }
    arrays, frame_count = _load_tracking(tracking)
    source_metadata = _load_tracking_metadata(tracking_metadata, tracking)
    asset_issues = sharpa_asset_blockers(robot_assets_dir)
    if asset_issues:
        raise AdapterError(
            "Sharpa asset validation failed: "
            + "; ".join(item["detail"] for item in asset_issues)
        )
    asset_snapshot = sharpa_asset_artifacts(robot_assets_dir)
    implementation_snapshot = _implementation_artifacts()
    trajectory_path = output_dir / "robot_trajectory.npz"
    metadata_path = output_dir / "robot_trajectory.json"
    existing = [path for path in (trajectory_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to replace: " + ", ".join(map(str, existing)))
    backend = ExistingSharpaBackend(
        device=device,
        robot_assets_dir=robot_assets_dir,
        max_frame_task_error_m=max_frame_task_error_m,
    )
    trajectory = trajectory_from_tracking(
        arrays,
        sharpa_backend=backend,
        mano_to_robot_scale=mano_to_robot_scale,
    )
    validity_mismatches: list[str] = []
    for side in ("left", "right"):
        input_valid = np.asarray(arrays[f"{side}_valid"], dtype=np.bool_)
        accepted_valid = np.asarray(trajectory[f"{side}_valid"], dtype=np.bool_)
        if not np.array_equal(accepted_valid, input_valid):
            rejected = np.flatnonzero(input_valid & ~accepted_valid)
            invented = np.flatnonzero(~input_valid & accepted_valid)
            validity_mismatches.append(
                f"{side}: accepted {int(np.count_nonzero(accepted_valid))}/"
                f"{int(np.count_nonzero(input_valid))} input-valid frames; "
                f"rejected indices={rejected.tolist()}, "
                f"unexpected accepted indices={invented.tolist()}"
            )
    if validity_mismatches:
        raise AdapterError(
            "Sharpa retargeting did not preserve every observed hand frame; "
            "refusing to commit a complete trajectory: "
            + "; ".join(validity_mismatches)
        )
    coordinate_frame = str(np.asarray(arrays["coordinate_frame"]).item())
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = trajectory_path.with_name(
        f".{trajectory_path.name}.{os.getpid()}.partial.npz"
    )
    try:
        np.savez_compressed(temporary, **trajectory)
        with np.load(temporary, allow_pickle=False) as archive:
            validate_robot_trajectory_arrays(dict(archive), expected_frames=frame_count)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    metadata: dict[str, Any] = {
        "schema_version": STAGE_SCHEMA,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "sequence_id": source_metadata.get("sequence_id"),
        "tracker": source_metadata.get("tracker"),
        "coordinate_frame": coordinate_frame,
        "robot": "dexmate_vega",
        "gripper": "sharpa_wave",
        "video": source_metadata.get("video") or source_metadata.get("geometry"),
        "sharpa_backend": backend.identity,
        "execution_environment": {
            "container_image_id": sharpa_image_id,
        },
        "implementation_sources": implementation_snapshot,
        "robot_assets": {
            "root": str(robot_assets_dir),
            "files": asset_snapshot,
        },
        "mano_to_robot_scale": float(mano_to_robot_scale),
        "max_frame_task_error_m": float(max_frame_task_error_m),
        "quality_gate_policy": {
            "metric": "maximum positional residual over Sharpa frame tasks",
            "acceptance": (
                "every input-valid frame must pass; source-invalid gaps remain invalid"
            ),
            "interpretation": (
                "catastrophic-solution rejection guard, not a convergence claim"
            ),
            "investigation_calibration": (
                "0.05 m rejected 25/74 finite observed frames per side on TACO "
                "sequence 20231005_253; 0.07 m preserves those observations while "
                "retaining explicit rejection and temporal-seed reset above the gate"
            ),
        },
        "quality": backend.diagnostics,
        "valid_frames": {
            side: int(np.count_nonzero(trajectory[f"{side}_valid"]))
            for side in ("left", "right")
        },
        "source_tracking": {
            "path": str(tracking),
            "size_bytes": tracking.stat().st_size,
            "sha256": _sha256(tracking),
            "metadata_path": str(tracking_metadata),
            "metadata_sha256": _sha256(tracking_metadata),
        },
        "robot_trajectory": {
            "path": str(trajectory_path),
            "size_bytes": temporary.stat().st_size,
            "sha256": _sha256(temporary),
        },
    }
    try:
        _assert_snapshot_unchanged(
            "tracking inputs",
            tracking_snapshot,
            {
                "tracking": _file_artifact(tracking),
                "tracking_metadata": _file_artifact(tracking_metadata),
            },
        )
        _assert_snapshot_unchanged(
            "Sharpa implementation sources",
            implementation_snapshot,
            _implementation_artifacts(),
        )
        _assert_snapshot_unchanged(
            "Sharpa XML/mesh assets",
            asset_snapshot,
            sharpa_asset_artifacts(robot_assets_dir),
        )
        if metadata_path.exists():
            metadata_path.unlink()
        temporary.replace(trajectory_path)
        _atomic_json(metadata_path, metadata)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", required=True, type=Path)
    parser.add_argument("--tracking-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--robot-assets-dir", type=Path, default=DEFAULT_ROBOT_ASSETS_DIR
    )
    parser.add_argument(
        "--sharpa-image-id",
        required=True,
        help=(
            "Immutable ID from docker image inspect robotic-grounding:latest "
            "--format '{{.Id}}'."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mano-to-robot-scale", type=float, default=1.2)
    parser.add_argument("--max-frame-task-error-m", type=float, default=0.07)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    common = {
        "tracking": args.tracking,
        "tracking_metadata": args.tracking_metadata,
        "output_dir": args.output_dir,
        "robot_assets_dir": args.robot_assets_dir,
        "sharpa_image_id": args.sharpa_image_id,
        "overwrite": args.overwrite,
    }
    if not args.execute:
        report = preflight_sharpa(**common)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["state"] == "ready" else 2)
    try:
        metadata = execute_sharpa(
            **common,
            device=args.device,
            mano_to_robot_scale=args.mano_to_robot_scale,
            max_frame_task_error_m=args.max_frame_task_error_m,
        )
    except (
        AdapterError,
        ContractError,
        FileExistsError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
