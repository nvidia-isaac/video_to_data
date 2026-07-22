"""Container-friendly MANO FK stage for the Video2Data inpainting condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from inpainting.adapters.video2data import (
    DEFAULT_MANO_MODEL_DIR,
    AdapterError,
    ManoTorchBackend,
    _atomic_json,
    _module_available,
    _sha256,
    default_sequence_id,
    load_result_bundle,
    load_wilor_json_bundle,
    tracking_from_bundle,
)
from inpainting.contracts import validate_tracking_arrays
from inpainting.video_io import probe_video


STAGE_SCHEMA = "v2d.inpainting.video2data-tracking-stage/v2"
WILOR_RUN_GENERATION_SCHEMA = "v2d.wilor.video-to-hands-generation/v1"
WILOR_RUN_GENERATION_FILENAME = "run_generation.json"
DEFAULT_PUBLIC_WEIGHTS_DIR = (
    Path(__file__).resolve().parents[1] / "artifacts" / "weights" / "wilor"
)
WILOR_SOURCE_COMMIT = "ebec42f94c389070cdd7dda6fd1bf0b4a659c960"
WILOR_HF_REPOSITORY = "warmshao/WiLoR-mini"
WILOR_HF_REVISION = "b00adea9a6843bbb4c9042109c5eb29ab2a59dea"
PUBLIC_WEIGHT_SHA256 = {
    "mano_mean_params.npz": "efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988",
    "wilor_final.ckpt": "3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2",
    "detector.pt": "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033",
}


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _aggregate_files(paths: list[Path]) -> dict[str, Any]:
    """Hash ordered names, sizes, and bytes without depending on root path."""

    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        name = path.name.encode("utf-8")
        size = path.stat().st_size
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        total_bytes += size
    return {
        "directory": str(paths[0].parent) if paths else None,
        "file_count": len(paths),
        "size_bytes": total_bytes,
        "aggregate_sha256": digest.hexdigest(),
        "aggregate_algorithm": (
            "sha256(concat(u64be(name_length),name_utf8,u64be(size),file_bytes) "
            "for lexical frame filenames)"
        ),
    }


def _artifact_identity(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "size_bytes": int(value["size_bytes"]),
        "sha256": str(value["sha256"]),
    }


def _wilor_static_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        bboxes = manifest["sources"]["bboxes"]
        return {
            "execution_environment": manifest["execution_environment"],
            "source_revisions": manifest["source_revisions"],
            "source_video": _artifact_identity(manifest["sources"]["video"]),
            "bboxes": (
                None
                if bboxes is None
                else {
                    name: _artifact_identity(artifact)
                    for name, artifact in sorted(bboxes["files"].items())
                }
            ),
            "weights": {
                name: _artifact_identity(artifact)
                for name, artifact in sorted(manifest["weights"].items())
            },
            "implementation_sources": {
                name: _artifact_identity(artifact)
                for name, artifact in sorted(manifest["implementation_sources"].items())
            },
            "parameters": manifest["parameters"],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise AdapterError(
            f"WiLoR run-generation manifest has incomplete identity: {exc}"
        ) from exc


def _wilor_generation_id(
    static_identity: dict[str, Any], frame_names: list[str]
) -> str:
    payload = {"identity": static_identity, "expected_frames": frame_names}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_wilor_run_generation(
    *,
    json_dir: Path,
    source_video: Path,
    expected_frames: int,
    expected_image_id: str,
) -> dict[str, Any]:
    """Validate the raw frame set against its atomic inference generation."""

    json_dir = json_dir.expanduser().resolve()
    manifest_path = json_dir / WILOR_RUN_GENERATION_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(
            f"Raw WiLoR input requires a valid {WILOR_RUN_GENERATION_FILENAME}: {exc}"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != WILOR_RUN_GENERATION_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise AdapterError("Raw WiLoR run-generation manifest is not complete v1")
    if (
        manifest.get("execution_environment", {}).get("container_image_id")
        != expected_image_id
    ):
        raise AdapterError("Raw WiLoR generation container image ID mismatch")
    if manifest.get("source_revisions") != {
        "wilor_mini_commit": WILOR_SOURCE_COMMIT,
        "huggingface_revision": WILOR_HF_REVISION,
    }:
        raise AdapterError("Raw WiLoR generation source revisions are not pinned")
    manifest_weights = manifest.get("weights")
    if not isinstance(manifest_weights, dict):
        raise AdapterError("Raw WiLoR generation omits consumed weight identities")
    for filename, expected_hash in PUBLIC_WEIGHT_SHA256.items():
        artifact = manifest_weights.get(filename)
        if not isinstance(artifact, dict) or artifact.get("sha256") != expected_hash:
            raise AdapterError(f"Raw WiLoR generation has unpinned weight {filename}")
    mano_artifact = manifest_weights.get("MANO_RIGHT.pkl")
    if (
        not isinstance(mano_artifact, dict)
        or not isinstance(mano_artifact.get("sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", mano_artifact["sha256"])
        or int(mano_artifact.get("size_bytes", 0)) <= 0
    ):
        raise AdapterError(
            "Raw WiLoR generation omits the licensed MANO_RIGHT.pkl identity"
        )
    video_artifact = manifest.get("sources", {}).get("video")
    if not isinstance(video_artifact, dict) or _artifact_identity(
        video_artifact
    ) != _artifact_identity(_artifact(source_video)):
        raise AdapterError("Raw WiLoR generation source video mismatch")
    expected = manifest.get("expected_frames")
    if not isinstance(expected, dict) or not isinstance(
        expected.get("filenames"), list
    ):
        raise AdapterError("Raw WiLoR generation omits its expected frame set")
    frame_names = expected["filenames"]
    canonical_names = [f"{index:06d}.json" for index in range(expected_frames)]
    if expected.get("count") != expected_frames or frame_names != canonical_names:
        raise AdapterError("Raw WiLoR generation frame set does not match the video")
    actual_entries = {path.name for path in json_dir.iterdir()}
    if actual_entries != {WILOR_RUN_GENERATION_FILENAME, *canonical_names}:
        raise AdapterError(
            "Raw WiLoR directory does not contain exactly its committed generation"
        )
    static_identity = _wilor_static_identity(manifest)
    if manifest.get("static_identity") != static_identity:
        raise AdapterError("Raw WiLoR generation static identity is invalid")
    if manifest.get("generation_id") != _wilor_generation_id(
        static_identity, canonical_names
    ):
        raise AdapterError("Raw WiLoR generation ID is invalid")

    paths = [json_dir / name for name in canonical_names]
    aggregate = _aggregate_files(paths)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise AdapterError("Raw WiLoR generation omits output hashes")
    for key in ("file_count", "size_bytes", "aggregate_sha256", "aggregate_algorithm"):
        if outputs.get(key) != aggregate.get(key):
            raise AdapterError(f"Raw WiLoR generation output {key} mismatch")
    output_files = outputs.get("files")
    if not isinstance(output_files, dict) or set(output_files) != set(canonical_names):
        raise AdapterError("Raw WiLoR generation per-frame hashes are incomplete")
    for path in paths:
        artifact = output_files[path.name]
        if not isinstance(artifact, dict) or _artifact_identity(
            artifact
        ) != _artifact_identity(_artifact(path)):
            raise AdapterError(
                f"Raw WiLoR frame hash no longer matches generation: {path.name}"
            )
    return {
        "manifest": _artifact(manifest_path),
        "generation_id": manifest["generation_id"],
        "container_image_id": expected_image_id,
        "expected_frames": expected,
        "outputs": outputs,
        "weights": manifest["weights"],
        "implementation_sources": manifest["implementation_sources"],
        "source_revisions": manifest["source_revisions"],
        "parameters": manifest["parameters"],
    }


def _public_weight_artifacts(
    weights_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    pretrained = weights_dir / "pretrained_models"
    artifacts: dict[str, Any] = {}
    blockers: list[dict[str, str]] = []
    for filename, expected in PUBLIC_WEIGHT_SHA256.items():
        path = pretrained / filename
        if not path.is_file():
            blockers.append(
                {
                    "code": "missing_public_wilor_weight",
                    "path": str(path),
                    "detail": f"Expected pinned {filename} for raw-output provenance.",
                }
            )
            continue
        info = _artifact(path)
        info["expected_sha256"] = expected
        artifacts[filename] = info
        if info["sha256"] != expected:
            blockers.append(
                {
                    "code": "public_wilor_weight_hash_mismatch",
                    "path": str(path),
                    "detail": f"{info['sha256']} != {expected}",
                }
            )
    return artifacts, blockers


def _load_input(
    *,
    result_dir: Path | None,
    wilor_json_dir: Path | None,
    taco_intrinsic: Path | None,
    taco_extrinsic: Path | None,
    source_video: Path,
    allow_static_camera: bool,
):
    geometry = probe_video(source_video)
    if (result_dir is None) == (wilor_json_dir is None):
        raise ValueError("Supply exactly one of result_dir or wilor_json_dir")
    if result_dir is not None:
        bundle = load_result_bundle(
            result_dir,
            expected_frames=geometry.frame_count,
            allow_static_camera=allow_static_camera,
        )
    else:
        if taco_intrinsic is None or taco_extrinsic is None:
            raise ValueError("Raw WiLoR input requires TACO intrinsic and extrinsic")
        bundle = load_wilor_json_bundle(
            wilor_json_dir,
            geometry=geometry,
            taco_intrinsic=taco_intrinsic,
            taco_extrinsic=taco_extrinsic,
        )
    return geometry, bundle


def preflight_tracking(
    *,
    result_dir: Path | None,
    wilor_json_dir: Path | None,
    taco_intrinsic: Path | None,
    taco_extrinsic: Path | None,
    source_video: Path,
    output_dir: Path,
    mano_model_dir: Path,
    public_weights_dir: Path,
    wilor_image_id: str | None,
    allow_static_camera: bool,
    overwrite: bool,
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    geometry = bundle = None
    raw_generation: dict[str, Any] | None = None
    source_video = source_video.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    mano_model_dir = mano_model_dir.expanduser().resolve()
    public_weights_dir = public_weights_dir.expanduser().resolve()
    if not source_video.is_file():
        blockers.append(
            {
                "code": "missing_source_video",
                "path": str(source_video),
                "detail": "Source video is required.",
            }
        )
    else:
        try:
            geometry, bundle = _load_input(
                result_dir=result_dir,
                wilor_json_dir=wilor_json_dir,
                taco_intrinsic=taco_intrinsic,
                taco_extrinsic=taco_extrinsic,
                source_video=source_video,
                allow_static_camera=allow_static_camera,
            )
        except (AdapterError, FileNotFoundError, ValueError) as exc:
            blockers.append(
                {
                    "code": "invalid_tracking_input",
                    "path": str(result_dir or wilor_json_dir),
                    "detail": str(exc),
                }
            )
    mano_files = [
        mano_model_dir / "models" / f"MANO_{side}.pkl" for side in ("LEFT", "RIGHT")
    ]
    for path in mano_files:
        if not path.is_file():
            blockers.append(
                {
                    "code": "missing_licensed_mano_model",
                    "path": str(path),
                    "detail": "Licensed MANO file is required read-only.",
                }
            )
    missing_modules = [
        name for name in ("torch", "manotorch") if not _module_available(name)
    ]
    if missing_modules:
        blockers.append(
            {
                "code": "missing_mano_dependencies",
                "path": "environment",
                "detail": "Missing: " + ", ".join(missing_modules),
            }
        )
    needs_wilor_provenance = wilor_json_dir is not None or (
        bundle is not None and bundle.hand_pose_source == "wilor"
    )
    public_weights: dict[str, Any] = {}
    if needs_wilor_provenance:
        public_weights, provenance_blockers = _public_weight_artifacts(
            public_weights_dir
        )
        blockers.extend(provenance_blockers)
        if wilor_image_id is None or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", wilor_image_id
        ):
            blockers.append(
                {
                    "code": "invalid_wilor_image_id",
                    "path": "--wilor-image-id",
                    "detail": "Pass the immutable sha256:<64 hex> Docker image ID.",
                }
            )
        elif wilor_json_dir is not None and geometry is not None:
            try:
                raw_generation = _validate_wilor_run_generation(
                    json_dir=wilor_json_dir,
                    source_video=source_video,
                    expected_frames=geometry.frame_count,
                    expected_image_id=wilor_image_id,
                )
            except (AdapterError, FileNotFoundError, OSError, ValueError) as exc:
                blockers.append(
                    {
                        "code": "invalid_wilor_run_generation",
                        "path": str(wilor_json_dir / WILOR_RUN_GENERATION_FILENAME),
                        "detail": str(exc),
                    }
                )
    outputs = [output_dir / "tracking.npz", output_dir / "tracking.json"]
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
        "source_video": str(source_video),
        "mano_files": [str(path) for path in mano_files],
        "public_weights_dir": str(public_weights_dir),
        "public_weights": public_weights,
        "wilor_image_id": wilor_image_id,
        "allow_static_camera": bool(allow_static_camera),
        "outputs": [str(path) for path in outputs],
        "blockers": blockers,
    }
    if geometry is not None:
        report["video"] = geometry.as_dict()
    if bundle is not None:
        report["input_mode"] = bundle.input_mode
        report["valid_frames"] = {
            side: int(np.count_nonzero(bundle.arrays[f"hand_{side}_is_valid"]))
            for side in ("left", "right")
        }
    if raw_generation is not None:
        report["wilor_run_generation"] = raw_generation
    return report


def execute_tracking(
    *,
    result_dir: Path | None,
    wilor_json_dir: Path | None,
    taco_intrinsic: Path | None,
    taco_extrinsic: Path | None,
    source_video: Path,
    output_dir: Path,
    mano_model_dir: Path,
    public_weights_dir: Path,
    wilor_image_id: str | None,
    device: str,
    sequence_id: str | None,
    allow_static_camera: bool,
    overwrite: bool,
) -> dict[str, Any]:
    source_video = source_video.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    mano_model_dir = mano_model_dir.expanduser().resolve()
    public_weights_dir = public_weights_dir.expanduser().resolve()
    geometry, bundle = _load_input(
        result_dir=result_dir,
        wilor_json_dir=wilor_json_dir,
        taco_intrinsic=taco_intrinsic,
        taco_extrinsic=taco_extrinsic,
        source_video=source_video,
        allow_static_camera=allow_static_camera,
    )
    tracking_path = output_dir / "tracking.npz"
    metadata_path = output_dir / "tracking.json"
    existing = [path for path in (tracking_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Refusing to replace: " + ", ".join(map(str, existing)))
    for path in (
        mano_model_dir / "models" / "MANO_LEFT.pkl",
        mano_model_dir / "models" / "MANO_RIGHT.pkl",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    public_weights: dict[str, Any] = {}
    raw_generation: dict[str, Any] | None = None
    if bundle.hand_pose_source == "wilor":
        if wilor_image_id is None or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", wilor_image_id
        ):
            raise ValueError("wilor_image_id must be immutable sha256:<64 hex>")
        public_weights, provenance_blockers = _public_weight_artifacts(
            public_weights_dir
        )
        if provenance_blockers:
            raise AdapterError(
                "WiLoR provenance validation failed: "
                + "; ".join(item["detail"] for item in provenance_blockers)
            )
        if bundle.input_mode == "raw_wilor_taco_camera":
            assert wilor_json_dir is not None
            raw_generation = _validate_wilor_run_generation(
                json_dir=wilor_json_dir,
                source_video=source_video,
                expected_frames=geometry.frame_count,
                expected_image_id=wilor_image_id,
            )
    backend = ManoTorchBackend(mano_model_dir, device=device)
    tracking = tracking_from_bundle(bundle, mano_backend=backend)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_temporary = tracking_path.with_name(
        f".{tracking_path.name}.{os.getpid()}.partial.npz"
    )
    try:
        np.savez_compressed(tracking_temporary, **tracking)
        with np.load(tracking_temporary, allow_pickle=False) as archive:
            validate_tracking_arrays(
                dict(archive), expected_frames=geometry.frame_count
            )
    except Exception:
        tracking_temporary.unlink(missing_ok=True)
        raise

    mano_files = {
        side.lower(): mano_model_dir / "models" / f"MANO_{side}.pkl"
        for side in ("LEFT", "RIGHT")
    }
    provenance: dict[str, Any] = {
        "source_video": _artifact(source_video),
        "mano_models": {side: _artifact(path) for side, path in mano_files.items()},
    }
    if bundle.hand_pose_source == "wilor":
        provenance["wilor"] = {
            "source_commit": WILOR_SOURCE_COMMIT,
            "huggingface_repository": WILOR_HF_REPOSITORY,
            "huggingface_revision": WILOR_HF_REVISION,
            "container_image_id": wilor_image_id,
            "public_weights": public_weights,
        }
    if bundle.input_mode == "raw_wilor_taco_camera":
        assert wilor_json_dir is not None
        assert taco_intrinsic is not None and taco_extrinsic is not None
        json_paths = [
            wilor_json_dir.expanduser().resolve() / f"{index:06d}.json"
            for index in range(geometry.frame_count)
        ]
        provenance["raw_wilor_json"] = _aggregate_files(json_paths)
        assert raw_generation is not None
        provenance["raw_wilor_generation"] = raw_generation
        provenance["taco_camera"] = {
            "intrinsic": _artifact(taco_intrinsic.expanduser().resolve()),
            "world_to_camera": _artifact(taco_extrinsic.expanduser().resolve()),
        }
    else:
        provenance["result_bundle"] = {
            "result_npz": _artifact(bundle.npz_path),
            "manifest": _artifact(bundle.manifest_path),
        }
    metadata: dict[str, Any] = {
        "schema_version": STAGE_SCHEMA,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "sequence_id": sequence_id or default_sequence_id(bundle.result_dir),
        "tracker": "v2d",
        "coordinate_frame": "world",
        "source_video": str(source_video),
        "video": geometry.as_dict(),
        "input_mode": bundle.input_mode,
        "input_files": [str(path) for path in bundle.input_files],
        "hand_pose_source": bundle.hand_pose_source,
        "mano_model_dir": str(mano_model_dir),
        "mano_backend": backend.identity,
        "provenance": provenance,
        "shape_policy": bundle.manifest.get("sources", {}).get("shape_policy"),
        "scale_policy": bundle.manifest.get("sources", {}).get(
            "hand_scale_policy", "bundle hand_scale"
        ),
        "valid_frames": {
            side: int(np.count_nonzero(tracking[f"{side}_valid"]))
            for side in ("left", "right")
        },
        "tracking": {
            "path": str(tracking_path),
            "size_bytes": tracking_temporary.stat().st_size,
            "sha256": _sha256(tracking_temporary),
        },
    }
    try:
        if raw_generation is not None:
            assert wilor_json_dir is not None
            current_generation = _validate_wilor_run_generation(
                json_dir=wilor_json_dir,
                source_video=source_video,
                expected_frames=geometry.frame_count,
                expected_image_id=str(wilor_image_id),
            )
            if current_generation != raw_generation:
                raise AdapterError(
                    "Raw WiLoR generation changed during MANO FK; refusing the "
                    "atomic tracking commit"
                )
        if metadata_path.exists():
            metadata_path.unlink()
        tracking_temporary.replace(tracking_path)
        _atomic_json(metadata_path, metadata)
    finally:
        tracking_temporary.unlink(missing_ok=True)
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--result-dir", type=Path)
    inputs.add_argument("--wilor-json-dir", type=Path)
    parser.add_argument("--taco-intrinsic", type=Path)
    parser.add_argument("--taco-extrinsic", type=Path)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mano-model-dir", type=Path, default=DEFAULT_MANO_MODEL_DIR)
    parser.add_argument(
        "--public-weights-dir", type=Path, default=DEFAULT_PUBLIC_WEIGHTS_DIR
    )
    parser.add_argument(
        "--wilor-image-id",
        help="Immutable ID from docker image inspect v2d_wilor:latest --format '{{.Id}}'.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sequence-id")
    parser.add_argument("--allow-static-camera", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    kwargs = {
        "result_dir": args.result_dir,
        "wilor_json_dir": args.wilor_json_dir,
        "taco_intrinsic": args.taco_intrinsic,
        "taco_extrinsic": args.taco_extrinsic,
        "source_video": args.source_video,
        "output_dir": args.output_dir,
        "mano_model_dir": args.mano_model_dir,
        "public_weights_dir": args.public_weights_dir,
        "wilor_image_id": args.wilor_image_id,
        "allow_static_camera": args.allow_static_camera,
        "overwrite": args.overwrite,
    }
    if not args.execute:
        report = preflight_tracking(**kwargs)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(0 if report["state"] == "ready" else 2)
    try:
        metadata = execute_tracking(
            **kwargs,
            device=args.device,
            sequence_id=args.sequence_id,
        )
    except (AdapterError, FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
