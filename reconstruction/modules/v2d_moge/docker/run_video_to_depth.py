# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run MoGe over a video and commit an auditable output generation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from v2d.docker.container import run_in_container
from v2d.moge.docker._config import IMAGE_NAME, MODULES_DIR


RUN_GENERATION_FILENAME = "run_generation.json"
RUN_GENERATION_SCHEMA = "v2d.moge.video-to-depth-generation/v1"
MOGE_REPOSITORY = "Ruicheng/moge-2-vitl-normal"
MOGE_REVISION = "b135031bae30b5ac2ae141a0e68717795ce38340"
MOGE_SOURCE_COMMIT = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_OUTPUT_EXTENSIONS = {
    "depth": ".png",
    "intrinsics": ".json",
    "points": ".npy",
    "normals": ".npy",
    "mask": ".png",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, recorded_path: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Generation input is not a regular file: {path}")
    return {
        "path": recorded_path if recorded_path is not None else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _artifact_identity(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "size_bytes": int(value["size_bytes"]),
        "sha256": str(value["sha256"]),
    }


def _model_checkpoint(weights_path: Path) -> Path:
    if weights_path.is_file():
        return weights_path.resolve()
    if not weights_path.is_dir():
        raise FileNotFoundError(f"MoGe weights path not found: {weights_path}")
    for name in ("model.pt", "pytorch_model.bin", "model.safetensors"):
        candidate = weights_path / name
        if candidate.is_file() and not candidate.is_symlink():
            return candidate.resolve()
    raise FileNotFoundError(
        "MoGe weights directory has no auditable model.pt, pytorch_model.bin, "
        f"or model.safetensors checkpoint: {weights_path}"
    )


def resolve_image_id(image: str = IMAGE_NAME) -> str:
    """Resolve a mutable local tag once, then execute its immutable image ID."""

    completed = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    image_id = completed.stdout.strip()
    if not _IMAGE_ID_PATTERN.fullmatch(image_id):
        raise RuntimeError(
            f"docker image inspect returned an invalid immutable ID for {image!r}: "
            f"{image_id!r}"
        )
    return image_id


def _implementation_sources() -> dict[str, dict[str, Any]]:
    module_root = Path(__file__).resolve().parents[1]
    paths = {
        "v2d/moge/docker/run_video_to_depth.py": Path(__file__),
        "v2d/moge/lib/video_to_depth.py": module_root / "lib/video_to_depth.py",
    }
    return {
        name: _artifact(path, recorded_path=name)
        for name, path in sorted(paths.items())
    }


def _source_snapshot(
    *,
    video_path: Path,
    input_intrinsics_path: Path | None,
    checkpoint_path: Path,
    image_id: str,
    batch_size: int,
    requested_outputs: list[str],
    dev: bool,
) -> dict[str, Any]:
    return {
        "execution_environment": {"container_image_id": image_id},
        "source_revisions": {
            "moge_repository": MOGE_REPOSITORY,
            "moge_huggingface_revision": MOGE_REVISION,
            "moge_source_commit": MOGE_SOURCE_COMMIT,
        },
        "sources": {
            "video": _artifact(video_path),
            "input_intrinsics": (
                None
                if input_intrinsics_path is None
                else _artifact(input_intrinsics_path)
            ),
        },
        "model": {"checkpoint": _artifact(checkpoint_path)},
        "implementation_sources": _implementation_sources(),
        "parameters": {
            "batch_size": batch_size,
            "input_intrinsics_path": (
                None
                if input_intrinsics_path is None
                else str(input_intrinsics_path.resolve())
            ),
            "intrinsics_mode": (
                "estimated_from_rgb"
                if input_intrinsics_path is None
                else "known_horizontal_fov_prior"
            ),
            "requested_outputs": requested_outputs,
            "dev_mount_enabled": dev,
            "frame_index_origin": 0,
            "depth_encoding": "uint16 PNG, encoded=65535/(depth_m+1)",
        },
    }


def _static_identity(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_environment": snapshot["execution_environment"],
        "source_revisions": snapshot["source_revisions"],
        "sources": {
            "video": _artifact_identity(snapshot["sources"]["video"]),
            "input_intrinsics": (
                None
                if snapshot["sources"]["input_intrinsics"] is None
                else _artifact_identity(snapshot["sources"]["input_intrinsics"])
            ),
        },
        "model": {
            "checkpoint": _artifact_identity(snapshot["model"]["checkpoint"])
        },
        "implementation_sources": {
            name: _artifact_identity(artifact)
            for name, artifact in sorted(snapshot["implementation_sources"].items())
        },
        "parameters": snapshot["parameters"],
    }


def _numbered_files(directory: Path, extension: str) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"MoGe output is not a regular directory: {directory}")
    entries = list(directory.iterdir())
    files = sorted(
        (path for path in entries if path.name != RUN_GENERATION_FILENAME),
        key=lambda path: path.name,
    )
    unexpected = [
        path
        for path in entries
        if path.name == RUN_GENERATION_FILENAME
        and (path.is_symlink() or not path.is_file())
    ]
    if unexpected:
        raise RuntimeError(
            f"MoGe generation marker is not a regular file: {unexpected[0]}"
        )
    if not files:
        raise RuntimeError(f"MoGe output directory is empty: {directory}")
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise RuntimeError(f"MoGe output contains a non-regular file: {directory}")
    expected = [f"{index:06d}{extension}" for index in range(len(files))]
    if [path.name for path in files] != expected:
        raise RuntimeError(
            f"MoGe output must contain exactly contiguous {extension} frames: "
            f"{directory}"
        )
    return files


def _directory_identity(paths: list[Path], *, directory: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files: dict[str, dict[str, Any]] = {}
    total_size = 0
    for path in paths:
        identity = _artifact_identity(_artifact(path))
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


def _outputs_snapshot(output_paths: dict[str, Path]) -> dict[str, Any]:
    files_by_output = {
        name: _numbered_files(path, _OUTPUT_EXTENSIONS[name])
        for name, path in output_paths.items()
    }
    frame_counts = {len(paths) for paths in files_by_output.values()}
    if len(frame_counts) != 1:
        raise RuntimeError("Every requested MoGe output must have the same frame count")
    return {
        name: _directory_identity(paths, directory=output_paths[name])
        for name, paths in sorted(files_by_output.items())
    }


def _outputs_identity(outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            key: value
            for key, value in output.items()
            if key != "directory"
        }
        for name, output in sorted(outputs.items())
    }


def _generation_id(
    static_identity: dict[str, Any], outputs: dict[str, Any]
) -> str:
    value = {
        "static_identity": static_identity,
        "outputs": _outputs_identity(outputs),
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _require_clean_outputs(
    output_paths: dict[str, Path], manifest_path: Path
) -> None:
    if os.path.lexists(manifest_path):
        raise FileExistsError(
            f"MoGe generation manifest already exists; refusing overwrite: "
            f"{manifest_path}"
        )
    resolved_outputs = [path.resolve() for path in output_paths.values()]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("Each MoGe output modality must use a distinct directory")
    for path in output_paths.values():
        if os.path.lexists(path):
            if path.is_symlink() or not path.is_dir():
                raise FileExistsError(f"MoGe output is not a directory: {path}")
            if any(path.iterdir()):
                raise FileExistsError(
                    "Existing MoGe output is not an empty uncommitted directory; "
                    f"refusing to mix generations: {path}"
                )


def validate_generation_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Validate one complete manifest, its generation ID, and all output bytes."""

    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid MoGe generation manifest: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != RUN_GENERATION_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise RuntimeError("MoGe generation manifest is not a complete v1 commit")
    try:
        recorded_static = _static_identity(manifest)
        outputs = manifest["outputs"]
        current_outputs = _outputs_snapshot(
            {name: Path(value["directory"]) for name, value in outputs.items()}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"MoGe generation manifest is incomplete: {exc}") from exc
    if manifest.get("static_identity") != recorded_static:
        raise RuntimeError("MoGe top-level provenance contradicts static_identity")
    if current_outputs != outputs:
        raise RuntimeError("MoGe output bytes no longer match the generation manifest")
    if manifest.get("generation_id") != _generation_id(recorded_static, outputs):
        raise RuntimeError("MoGe generation ID is invalid")
    return manifest


def run_video_to_depth(
    video_path: str,
    depth_folder: str,
    intrinsics_folder: str,
    weights_path: str,
    batch_size: int = 8,
    input_intrinsics_path: str | None = None,
    points_folder: str | None = None,
    normals_folder: str | None = None,
    mask_folder: str | None = None,
    dev: bool = False,
    gpu: int = 0,
    *,
    generation_manifest_path: str | None = None,
    image_id: str | None = None,
) -> dict[str, Any]:
    """Run one fresh MoGe generation and atomically publish its commit marker."""

    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise ValueError("gpu must be a non-negative physical GPU index")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    resolved_image_id = image_id or resolve_image_id()
    if not _IMAGE_ID_PATTERN.fullmatch(resolved_image_id):
        raise ValueError(
            "image_id must be an immutable sha256:<64 lowercase hex> Docker ID"
        )

    video = Path(video_path).resolve()
    weights = Path(weights_path).resolve()
    input_intrinsics = (
        None
        if input_intrinsics_path is None
        else Path(input_intrinsics_path).resolve()
    )
    checkpoint = _model_checkpoint(weights)
    if not video.is_file():
        raise FileNotFoundError(f"MoGe source video not found: {video}")
    if input_intrinsics is not None and not input_intrinsics.is_file():
        raise FileNotFoundError(
            f"MoGe input intrinsics not found: {input_intrinsics}"
        )

    output_paths = {
        name: Path(value).resolve()
        for name, value in {
            "depth": depth_folder,
            "intrinsics": intrinsics_folder,
            "points": points_folder,
            "normals": normals_folder,
            "mask": mask_folder,
        }.items()
        if value is not None
    }
    manifest_path = Path(
        generation_manifest_path
        if generation_manifest_path is not None
        else Path(depth_folder) / RUN_GENERATION_FILENAME
    ).resolve()
    _require_clean_outputs(output_paths, manifest_path)

    requested_outputs = sorted(output_paths)
    initial_snapshot = _source_snapshot(
        video_path=video,
        input_intrinsics_path=input_intrinsics,
        checkpoint_path=checkpoint,
        image_id=resolved_image_id,
        batch_size=batch_size,
        requested_outputs=requested_outputs,
        dev=dev,
    )
    static_identity = _static_identity(initial_snapshot)

    inputs = {"video_path": str(video), "weights_path": str(weights)}
    input_files = {"video_path"}
    if input_intrinsics is not None:
        inputs["input_intrinsics_path"] = str(input_intrinsics)
        input_files.add("input_intrinsics_path")
    outputs = {
        f"{name}_folder": str(path) for name, path in output_paths.items()
    }
    run_in_container(
        image=resolved_image_id,
        module="v2d.moge.lib.video_to_depth",
        inputs=inputs,
        outputs=outputs,
        extra_args={"batch_size": batch_size},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpu_device=gpu,
        env={"CUDA_VISIBLE_DEVICES": "0"},
        input_directories={"weights_path"},
        input_files=input_files,
        output_directories=set(outputs),
    )

    final_snapshot = _source_snapshot(
        video_path=video,
        input_intrinsics_path=input_intrinsics,
        checkpoint_path=checkpoint,
        image_id=resolved_image_id,
        batch_size=batch_size,
        requested_outputs=requested_outputs,
        dev=dev,
    )
    if _static_identity(final_snapshot) != static_identity:
        raise RuntimeError(
            "MoGe source, intrinsics, model, or implementation changed during "
            "inference; refusing generation commit"
        )
    output_snapshot = _outputs_snapshot(output_paths)
    frame_count = next(iter(output_snapshot.values()))["file_count"]
    manifest = {
        "schema_version": RUN_GENERATION_SCHEMA,
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **initial_snapshot,
        "static_identity": static_identity,
        "invocation": {
            "container_module": "v2d.moge.lib.video_to_depth",
            "dev_mount_enabled": dev,
            "physical_gpu_index": gpu,
            "host_outputs": {
                name: str(path) for name, path in sorted(output_paths.items())
            },
            "generation_manifest_path": str(manifest_path),
        },
        "expected_frames": {
            "count": frame_count,
            "indices": [0, frame_count - 1],
        },
        "outputs": output_snapshot,
        "generation_id": _generation_id(static_identity, output_snapshot),
    }
    _atomic_json(manifest_path, manifest)
    try:
        return validate_generation_manifest(manifest_path)
    except Exception:
        manifest_path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--depth_folder", type=str, required=True)
    parser.add_argument("--intrinsics_folder", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_intrinsics_path", type=str, default=None)
    parser.add_argument("--points_folder", type=str, default=None)
    parser.add_argument("--normals_folder", type=str, default=None)
    parser.add_argument("--mask_folder", type=str, default=None)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--gpu", type=int, default=0, help="Physical host GPU index")
    parser.add_argument(
        "--generation_manifest_path",
        default=None,
        help=(
            "Atomic generation commit path; defaults to "
            "<depth_folder>/run_generation.json"
        ),
    )
    parser.add_argument(
        "--image_id",
        default=None,
        help=(
            "Optional immutable Docker ID; defaults to resolving the local "
            f"{IMAGE_NAME!r} tag once"
        ),
    )
    args = parser.parse_args()
    run_video_to_depth(
        args.video_path,
        args.depth_folder,
        args.intrinsics_folder,
        args.weights_path,
        batch_size=args.batch_size,
        input_intrinsics_path=args.input_intrinsics_path,
        points_folder=args.points_folder,
        normals_folder=args.normals_folder,
        mask_folder=args.mask_folder,
        dev=args.dev,
        gpu=args.gpu,
        generation_manifest_path=args.generation_manifest_path,
        image_id=args.image_id,
    )
