"""Shared host-side Docker command construction for the Isaac Sim workflows."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


IMAGE_NAME = os.environ.get("V2D_HOI_MESH_TO_USD_IMAGE", "v2d_hoi_mesh_to_usd")
MODULE_DIR = Path(__file__).resolve().parent
CONTAINER_MODULE_DIR = "/workspace/mesh_to_usd"
CONTAINER_INPUT_DIR = "/data/input"
CONTAINER_OUTPUT_DIR = "/data/output"
SUPPORTED_INPUTS = frozenset(
    {".usd", ".usda", ".usdc", ".obj", ".fbx", ".gltf", ".glb", ".stl"}
)

_CACHE_MOUNTS = (
    ("cache", "/tmp/.cache"),
    ("computecache", "/tmp/.nv/ComputeCache"),
    ("logs", "/tmp/.nvidia-omniverse/logs"),
    ("config", "/tmp/.nvidia-omniverse/config"),
    ("data", "/tmp/.local/share/ov/data"),
    ("pkg", "/tmp/.local/share/ov/pkg"),
    ("kit_cache", "/isaac-sim/kit/cache"),
    ("kit_data", "/isaac-sim/kit/data"),
    ("kit_logs", "/isaac-sim/kit/logs"),
)


def isaac_sim_eula_accepted(accept_eula: bool = False) -> bool:
    """Return whether Isaac Sim EULA acceptance was explicitly provided."""

    return accept_eula or os.environ.get("ACCEPT_EULA", "").upper() == "Y"


def build_isaac_sim_command(
    asset_path: str,
    output_dir: str,
    *,
    cache_dir: str | None,
    accept_eula: bool,
    dev: bool,
    gpu_device: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, list[str]]:
    """Validate host paths and build the common Docker command prefix."""

    asset = Path(asset_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not asset.is_file() or asset.suffix.lower() not in SUPPORTED_INPUTS:
        raise ValueError(f"Unsupported or missing asset: {asset}")
    if not isaac_sim_eula_accepted(accept_eula):
        raise ValueError(
            "The Isaac Sim container requires EULA acceptance. Set ACCEPT_EULA=Y "
            "or pass --accept-eula after reviewing the NVIDIA Isaac Sim EULA."
        )
    if gpu_device is not None:
        gpu_device = str(gpu_device).strip()
        if not gpu_device or "," in gpu_device or any(
            character.isspace() for character in gpu_device
        ):
            raise ValueError("gpu_device must identify exactly one GPU")

    output.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir or "~/.cache/v2d/isaac-sim").expanduser().resolve()
    for name, _destination in _CACHE_MOUNTS:
        (cache / name).mkdir(parents=True, exist_ok=True)

    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all" if gpu_device is None else f"device={gpu_device}",
        "--shm-size=2g",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--group-add",
        "1234",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "HOME=/tmp",
        "-v",
        f"{asset.parent}:{CONTAINER_INPUT_DIR}:ro",
        "-v",
        f"{output}:{CONTAINER_OUTPUT_DIR}",
    ]
    for name, destination in _CACHE_MOUNTS:
        command.extend(["-v", f"{cache / name}:{destination}"])
    for name, value in (environment or {}).items():
        command.extend(["-e", f"{name}={value}"])
    if dev:
        command.extend(["-v", f"{MODULE_DIR}:{CONTAINER_MODULE_DIR}:ro"])
    return asset, command
