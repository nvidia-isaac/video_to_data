"""Build or execute the offline renderer container command.

The safe default prints the complete command.  ``--execute`` is required to
start a container, and GPU access is granted only when ``--gpu`` is explicit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np


DEFAULT_IMAGE = "robotic-grounding:photo-render-v6"
_GPU_INDEX = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_GPU_UUID = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\Z"
)


@dataclass(frozen=True)
class ContainerConfig:
    trajectory: Path
    intrinsics: Path
    world_to_camera: Path
    asset_root: Path
    scene_utils_root: Path
    output_dir: Path
    width: int
    height: int
    fps: float
    repository_root: Path
    image_id: str
    image: str = DEFAULT_IMAGE
    arm_center_world: Path | None = None
    background_rgb: str = "0,0,0"
    max_ik_residual_m: float = 0.01
    max_joint_step_rad: float = 0.4
    gpu: str | None = None
    dry_run: bool = False
    overwrite: bool = False


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
    """Build a Docker ``--volume`` file mount.

    Docker's long ``--mount`` syntax treats commas inside a source path as
    option separators.  Official TACO directories are named like
    ``(dust, brush, cup)``, so use the short bind syntax for individual files.
    A colon is the only ambiguous character in that syntax on Linux and is
    rejected explicitly rather than mis-mounted.
    """

    source_text = str(source)
    if ":" in source_text:
        raise ValueError(f"Docker bind source must not contain ':': {source_text}")
    return f"{source_text}:{destination}:ro"


def resolve_local_image_id(image: str) -> str:
    """Resolve one human-readable local image reference to its immutable ID."""

    if not image.strip():
        raise ValueError("container image must not be empty")
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"cannot resolve local Docker image {image!r}: {detail}")
    image_id = completed.stdout.strip()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise RuntimeError(
            f"docker returned invalid immutable image ID for {image!r}: {image_id!r}"
        )
    return image_id


def validate_gpu_selector(value: str | None) -> str | None:
    """Accept exactly one GPU index or one canonical NVIDIA GPU UUID."""

    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("--gpu must name exactly one GPU index or GPU UUID")
    if _GPU_INDEX.fullmatch(value) is None and _GPU_UUID.fullmatch(value) is None:
        raise ValueError(
            "--gpu must name exactly one non-negative index or canonical GPU UUID; "
            "lists, 'all', whitespace, and partial UUIDs are forbidden"
        )
    return value


def build_docker_command(config: ContainerConfig) -> list[str]:
    """Return a mount-explicit Docker command without changing host state."""

    trajectory = _require_file(config.trajectory, "trajectory")
    intrinsics = _require_file(config.intrinsics, "intrinsics")
    world_to_camera = _require_file(config.world_to_camera, "world-to-camera")
    asset_root = _require_directory(config.asset_root, "asset root")
    scene_utils_root = _require_directory(config.scene_utils_root, "scene-utils root")
    repository_root = _require_directory(config.repository_root, "repository root")
    arm_center = (
        _require_file(config.arm_center_world, "arm-center transform")
        if config.arm_center_world is not None
        else None
    )
    output_dir = config.output_dir.resolve()
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"output parent: {output_dir.parent}")
    if not config.image.strip():
        raise ValueError("container image must not be empty")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", config.image_id) is None:
        raise ValueError("container image ID must be sha256: followed by 64 lowercase hex digits")
    if config.width <= 0 or config.height <= 0:
        raise ValueError("width and height must be positive")
    if not np.isfinite(config.fps) or config.fps <= 0.0:
        raise ValueError("fps must be positive and finite")

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
    if config.gpu is not None:
        command.extend(("--gpus", f"device={validate_gpu_selector(config.gpu)}"))
    command.extend(
        (
            "--mount",
            f"type=bind,src={repository_root},dst=/workspace/video_to_data,readonly",
            "--volume",
            _read_only_volume(trajectory, "/inputs/robot_trajectory.npz"),
            "--volume",
            _read_only_volume(intrinsics, f"/inputs/intrinsic{intrinsics.suffix}"),
            "--volume",
            _read_only_volume(
                world_to_camera, f"/inputs/world_to_camera{world_to_camera.suffix}"
            ),
            "--mount",
            f"type=bind,src={asset_root},dst=/external_assets,readonly",
            "--mount",
            f"type=bind,src={scene_utils_root},dst=/external_scene_utils,readonly",
            "--mount",
            f"type=bind,src={output_dir},dst=/output",
        )
    )
    if arm_center is not None:
        command.extend(
            (
                "--volume",
                _read_only_volume(arm_center, f"/inputs/arm_center{arm_center.suffix}"),
            )
        )
    command.extend(
        (
            "--entrypoint",
            "/workspace/isaaclab/isaaclab.sh",
            config.image_id,
            "-p",
            "-m",
            "inpainting.robot_renderer.cli",
            "--trajectory",
            "/inputs/robot_trajectory.npz",
            "--intrinsics",
            f"/inputs/intrinsic{intrinsics.suffix}",
            "--world-to-camera",
            f"/inputs/world_to_camera{world_to_camera.suffix}",
            "--width",
            str(config.width),
            "--height",
            str(config.height),
            "--fps",
            f"{config.fps:.12g}",
            "--asset-root",
            "/external_assets",
            "--scene-utils-root",
            "/external_scene_utils",
            "--output-dir",
            "/output",
            "--background-rgb",
            config.background_rgb,
            "--max-ik-residual-m",
            f"{config.max_ik_residual_m:.12g}",
            "--max-joint-step-rad",
            f"{config.max_joint_step_rad:.12g}",
        )
    )
    if arm_center is not None:
        command.extend(("--arm-center-world", f"/inputs/arm_center{arm_center.suffix}"))
    if config.dry_run:
        command.append("--dry-run")
    if config.overwrite:
        command.append("--overwrite")
    return command


def build_parser() -> argparse.ArgumentParser:
    package_file = Path(__file__).resolve()
    repository_default = package_file.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--world-to-camera", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--scene-utils-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--repository-root", type=Path, default=repository_default)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--arm-center-world", type=Path)
    parser.add_argument("--background-rgb", default="0,0,0")
    parser.add_argument("--max-ik-residual-m", type=float, default=0.01)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.4)
    parser.add_argument(
        "--gpu",
        help="Docker GPU device selector (for example 0 or GPU-UUID); omitted means no GPU.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create output directory and run Docker; default only prints the command.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_id = resolve_local_image_id(args.image)
    config = ContainerConfig(
        trajectory=args.trajectory,
        intrinsics=args.intrinsics,
        world_to_camera=args.world_to_camera,
        asset_root=args.asset_root,
        scene_utils_root=args.scene_utils_root,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        repository_root=args.repository_root,
        image_id=image_id,
        image=args.image,
        arm_center_world=args.arm_center_world,
        background_rgb=args.background_rgb,
        max_ik_residual_m=args.max_ik_residual_m,
        max_joint_step_rad=args.max_joint_step_rad,
        gpu=args.gpu,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    if args.execute:
        config.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    command = build_docker_command(config)
    print(shlex.join(command), flush=True)
    if not args.execute:
        return 0
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
