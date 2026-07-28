"""Build or execute the pinned offline TACO object-depth container command."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import subprocess
import uuid

from .robot_renderer.container_runner import (
    resolve_local_image_id,
    validate_gpu_selector,
)


DEFAULT_IMAGE = "robotic-grounding:photo-render-v6"


@dataclass(frozen=True)
class ObjectDepthContainerConfig:
    sequence_id: str
    parquet: Path
    source_video: Path
    intrinsics: Path
    world_to_camera: Path
    mesh_root: Path
    output_dir: Path
    repository_root: Path
    image_id: str
    image: str = DEFAULT_IMAGE
    gpu: str | None = None
    dry_run: bool = False
    overwrite: bool = False
    container_name: str | None = None


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label}: {resolved}")
    return resolved


def _require_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label}: {resolved}")
    return resolved


def _read_only_volume(source: Path, destination: str) -> str:
    source_text = str(source)
    if ":" in source_text:
        raise ValueError(f"Docker bind source must not contain ':': {source_text}")
    return f"{source_text}:{destination}:ro"


def build_docker_command(config: ObjectDepthContainerConfig) -> list[str]:
    parquet = _require_file(config.parquet, "TACO parquet")
    if not config.sequence_id or config.sequence_id.strip() != config.sequence_id:
        raise ValueError("sequence_id must be a non-empty, whitespace-trimmed string")
    partition_ids = [
        part.removeprefix("sequence_id=")
        for part in parquet.parts
        if part.startswith("sequence_id=")
    ]
    if len(partition_ids) != 1 or partition_ids[0] != config.sequence_id:
        raise ValueError(
            f"Explicit sequence_id {config.sequence_id!r} does not match exactly one "
            f"Hive partition in {parquet}: {partition_ids}"
        )
    source_video = _require_file(config.source_video, "source video")
    intrinsics = _require_file(config.intrinsics, "intrinsics")
    world_to_camera = _require_file(config.world_to_camera, "world-to-camera")
    mesh_root = _require_directory(config.mesh_root, "TACO mesh root")
    repository_root = _require_directory(config.repository_root, "repository root")
    output_dir = config.output_dir.resolve()
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"output parent: {output_dir.parent}")
    if not config.image.strip():
        raise ValueError("container image must not be empty")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", config.image_id) is None:
        raise ValueError(
            "container image ID must be sha256: followed by 64 lowercase hex digits"
        )

    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--shm-size",
        "2g",
        "--workdir",
        "/workspace/video_to_data",
        "--env",
        "PYOPENGL_PLATFORM=egl",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        f"V2D_RENDER_CONTAINER_IMAGE={config.image}",
        "--env",
        f"V2D_RENDER_CONTAINER_IMAGE_ID={config.image_id}",
        "--env",
        f"V2D_RENDER_HOST_OUTPUT_DIR={output_dir}",
        "--env",
        f"V2D_RENDER_HOST_UID={os.getuid()}",
        "--env",
        f"V2D_RENDER_HOST_GID={os.getgid()}",
    ]
    if config.container_name is not None:
        if not config.container_name or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for character in config.container_name
        ):
            raise ValueError(
                "container_name contains unsupported Docker name characters"
            )
        command.extend(("--name", config.container_name))
    if config.gpu is not None:
        command.extend(("--gpus", f"device={validate_gpu_selector(config.gpu)}"))
    command.extend(
        (
            "--mount",
            f"type=bind,src={repository_root},dst=/workspace/video_to_data,readonly",
            "--volume",
            _read_only_volume(parquet, "/inputs/motion.parquet"),
            "--volume",
            _read_only_volume(source_video, "/inputs/source.mp4"),
            "--volume",
            _read_only_volume(intrinsics, f"/inputs/intrinsic{intrinsics.suffix}"),
            "--volume",
            _read_only_volume(
                world_to_camera, f"/inputs/world_to_camera{world_to_camera.suffix}"
            ),
            "--mount",
            f"type=bind,src={mesh_root},dst=/external_taco_meshes,readonly",
            "--mount",
            f"type=bind,src={output_dir},dst=/output",
            "--entrypoint",
            "/workspace/isaaclab/isaaclab.sh",
            config.image_id,
            "-p",
            "-m",
            "inpainting.taco_object_depth",
            "--sequence-id",
            config.sequence_id,
            "--parquet",
            "/inputs/motion.parquet",
            "--source-video",
            "/inputs/source.mp4",
            "--intrinsics",
            f"/inputs/intrinsic{intrinsics.suffix}",
            "--world-to-camera",
            f"/inputs/world_to_camera{world_to_camera.suffix}",
            "--mesh-root",
            "/external_taco_meshes",
            "--output-dir",
            "/output",
            "--source-parquet-recorded-path",
            str(parquet),
            "--source-video-recorded-path",
            str(source_video),
            "--intrinsics-recorded-path",
            str(intrinsics),
            "--world-to-camera-recorded-path",
            str(world_to_camera),
        )
    )
    if config.dry_run:
        command.append("--dry-run")
    if config.overwrite:
        command.append("--overwrite")
    return command


def build_parser() -> argparse.ArgumentParser:
    repository_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--source-video", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--world-to-camera", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=repository_default)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--gpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.execute and not args.dry_run and args.gpu is None:
        parser.error("full EGL rendering requires an explicit --gpu device selector")
    if not args.timeout_seconds > 0:
        parser.error("--timeout-seconds must be positive")
    image_id = resolve_local_image_id(args.image)
    if args.execute:
        args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    container_name = f"v2d-taco-object-{uuid.uuid4().hex[:12]}"
    config = ObjectDepthContainerConfig(
        sequence_id=args.sequence_id,
        parquet=args.parquet,
        source_video=args.source_video,
        intrinsics=args.intrinsics,
        world_to_camera=args.world_to_camera,
        mesh_root=args.mesh_root,
        output_dir=args.output_dir,
        repository_root=args.repository_root,
        image_id=image_id,
        image=args.image,
        gpu=args.gpu,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        container_name=container_name,
    )
    command = build_docker_command(config)
    print(shlex.join(command), flush=True)
    if not args.execute:
        return 0
    try:
        completed = subprocess.run(
            command, check=False, timeout=float(args.timeout_seconds)
        )
        return int(completed.returncode)
    except subprocess.TimeoutExpired:
        subprocess.run(
            ["docker", "stop", "--timeout", "10", container_name],
            check=False,
            timeout=30,
        )
        print(
            f"Object-depth container exceeded {args.timeout_seconds:g}s and was stopped: "
            f"{container_name}"
        )
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
