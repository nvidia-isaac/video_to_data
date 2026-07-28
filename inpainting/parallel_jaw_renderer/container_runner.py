"""Build or execute a mount-explicit offline parallel-jaw render container."""

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
    target: Path
    bundle: Path
    intrinsics: Path
    world_to_camera: Path
    robot_asset_root: Path
    scene_utils_root: Path
    output_dir: Path
    width: int
    height: int
    fps: float
    repository_root: Path
    image_id: str
    image: str = DEFAULT_IMAGE
    T_world_hub: tuple[float, ...] | None = None
    T_world_hub_metadata: Path | None = None
    background_rgb: str = "0,0,0"
    max_ik_residual_m: float = 0.01
    ik_orientation_cost: float = 0.010
    max_orientation_residual_deg: float = 20.0
    max_joint_step_rad: float = 0.4
    gpu: str | None = None
    dry_run: bool = False
    overwrite: bool = False
    preview_frame_index: int | None = None


def _require_file(path: Path, label: str) -> Path:
    result = path.resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label}: {result}")
    return result


def _require_directory(path: Path, label: str) -> Path:
    result = path.resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"{label}: {result}")
    return result


def _volume(source: Path, destination: str) -> str:
    source_text = str(source)
    if ":" in source_text:
        raise ValueError(f"Docker bind source must not contain ':': {source_text}")
    return f"{source_text}:{destination}:ro"


def validate_gpu_selector(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or (_GPU_INDEX.fullmatch(value) is None and _GPU_UUID.fullmatch(value) is None)
    ):
        raise ValueError("--gpu must be one non-negative index or canonical GPU UUID")
    return value


def resolve_local_image_id(image: str) -> str:
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
        raise RuntimeError(f"cannot resolve local image {image!r}: {detail}")
    image_id = completed.stdout.strip()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise RuntimeError(f"docker returned invalid image ID {image_id!r}")
    return image_id


def build_docker_command(config: ContainerConfig) -> list[str]:
    target = _require_file(config.target, "target")
    bundle = _require_file(config.bundle, "bundle")
    intrinsics = _require_file(config.intrinsics, "intrinsics")
    world_to_camera = _require_file(config.world_to_camera, "world-to-camera")
    robot_asset_root = _require_directory(config.robot_asset_root, "robot asset root")
    scene_utils_root = _require_directory(config.scene_utils_root, "scene-utils root")
    repository_root = _require_directory(config.repository_root, "repository root")
    if (config.T_world_hub is None) == (config.T_world_hub_metadata is None):
        raise ValueError(
            "exactly one of T_world_hub or T_world_hub_metadata is required"
        )
    hub_metadata = (
        _require_file(config.T_world_hub_metadata, "T_world_hub metadata")
        if config.T_world_hub_metadata is not None
        else None
    )
    if config.T_world_hub is not None:
        values = np.asarray(config.T_world_hub, dtype=np.float64)
        if values.shape != (16,) or not np.isfinite(values).all():
            raise ValueError("T_world_hub must contain 16 finite row-major values")
    output_dir = config.output_dir.resolve()
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(f"output parent: {output_dir.parent}")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", config.image_id) is None:
        raise ValueError("image_id must be an immutable sha256 ID")
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
            _volume(target, "/inputs/parallel_jaw_target.npz"),
            "--volume",
            _volume(intrinsics, f"/inputs/intrinsic{intrinsics.suffix}"),
            "--volume",
            _volume(
                world_to_camera,
                f"/inputs/world_to_camera{world_to_camera.suffix}",
            ),
            "--mount",
            f"type=bind,src={bundle.parent},dst=/robot_bundle,readonly",
            "--mount",
            f"type=bind,src={robot_asset_root},dst=/robot_assets,readonly",
            "--mount",
            f"type=bind,src={scene_utils_root},dst=/external_scene_utils,readonly",
            "--mount",
            f"type=bind,src={output_dir},dst=/output",
        )
    )
    if hub_metadata is not None:
        command.extend(
            (
                "--volume",
                _volume(hub_metadata, "/inputs/T_world_hub_metadata.json"),
            )
        )
    command.extend(
        (
            "--entrypoint",
            "/workspace/isaaclab/isaaclab.sh",
            config.image_id,
            "-p",
            "-m",
            "inpainting.parallel_jaw_renderer.cli",
            "--target",
            "/inputs/parallel_jaw_target.npz",
            "--bundle",
            f"/robot_bundle/{bundle.name}",
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
            "--scene-utils-root",
            "/external_scene_utils",
            "--output-dir",
            "/output",
            "--background-rgb",
            config.background_rgb,
            "--max-ik-residual-m",
            f"{config.max_ik_residual_m:.12g}",
            "--ik-orientation-cost",
            f"{config.ik_orientation_cost:.12g}",
            "--max-orientation-residual-deg",
            f"{config.max_orientation_residual_deg:.12g}",
            "--max-joint-step-rad",
            f"{config.max_joint_step_rad:.12g}",
        )
    )
    if hub_metadata is not None:
        command.extend(("--T-world-hub-metadata", "/inputs/T_world_hub_metadata.json"))
    else:
        command.append("--T-world-hub")
        command.extend(f"{value:.17g}" for value in config.T_world_hub or ())
    if config.dry_run:
        command.append("--dry-run")
    if config.preview_frame_index is not None:
        if config.preview_frame_index < 0:
            raise ValueError("preview_frame_index must be non-negative")
        command.extend(("--preview-frame-index", str(config.preview_frame_index)))
    if config.overwrite:
        command.append("--overwrite")
    return command


def build_parser() -> argparse.ArgumentParser:
    repository_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--intrinsics", required=True, type=Path)
    parser.add_argument("--world-to-camera", required=True, type=Path)
    parser.add_argument("--robot-asset-root", required=True, type=Path)
    parser.add_argument("--scene-utils-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--repository-root", type=Path, default=repository_default)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    hub = parser.add_mutually_exclusive_group(required=True)
    hub.add_argument("--T-world-hub", dest="T_world_hub", nargs=16, type=float)
    hub.add_argument(
        "--T-world-hub-metadata",
        dest="T_world_hub_metadata",
        type=Path,
    )
    parser.add_argument("--background-rgb", default="0,0,0")
    parser.add_argument("--max-ik-residual-m", type=float, default=0.01)
    parser.add_argument("--ik-orientation-cost", type=float, default=0.010)
    parser.add_argument("--max-orientation-residual-deg", type=float, default=20.0)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.4)
    parser.add_argument("--gpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-frame-index", type=int)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Create the output directory and execute; default prints the command.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_id = resolve_local_image_id(args.image)
    config = ContainerConfig(
        target=args.target,
        bundle=args.bundle,
        intrinsics=args.intrinsics,
        world_to_camera=args.world_to_camera,
        robot_asset_root=args.robot_asset_root,
        scene_utils_root=args.scene_utils_root,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        repository_root=args.repository_root,
        image_id=image_id,
        image=args.image,
        T_world_hub=(
            tuple(float(value) for value in args.T_world_hub)
            if args.T_world_hub is not None
            else None
        ),
        T_world_hub_metadata=args.T_world_hub_metadata,
        background_rgb=args.background_rgb,
        max_ik_residual_m=args.max_ik_residual_m,
        ik_orientation_cost=args.ik_orientation_cost,
        max_orientation_residual_deg=args.max_orientation_residual_deg,
        max_joint_step_rad=args.max_joint_step_rad,
        gpu=args.gpu,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        preview_frame_index=args.preview_frame_index,
    )
    if args.execute:
        config.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    command = build_docker_command(config)
    print(shlex.join(command), flush=True)
    if not args.execute:
        return 0
    return int(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
