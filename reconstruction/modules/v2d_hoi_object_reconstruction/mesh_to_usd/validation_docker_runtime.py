"""Host-side Docker command construction for standalone USD validation."""

from __future__ import annotations

import os
from pathlib import Path


VALIDATOR_IMAGE_NAME = os.environ.get(
    "V2D_HOI_MESH_TO_USD_VALIDATOR_IMAGE",
    "v2d_hoi_mesh_to_usd_validator",
)
MODULE_DIR = Path(__file__).resolve().parent
CONTAINER_MODULE_DIR = "/workspace/mesh_to_usd"
CONTAINER_INPUT_DIR = "/data/input"
CONTAINER_OUTPUT_DIR = "/data/output"
VALIDATION_REPORT_NAME = "simready_validation_report.json"
SUPPORTED_INPUTS = frozenset({".usd", ".usda"})


def build_validator_command(
    asset_path: str,
    output_dir: str,
    *,
    image: str = VALIDATOR_IMAGE_NAME,
    dev: bool = False,
) -> tuple[Path, list[str]]:
    """Validate paths and build the standalone validator Docker command."""

    asset = Path(asset_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not asset.is_file() or asset.suffix.lower() not in SUPPORTED_INPUTS:
        raise ValueError(f"Unsupported or missing USD asset: {asset}")
    output.mkdir(parents=True, exist_ok=True)

    command = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-v",
        f"{asset.parent}:{CONTAINER_INPUT_DIR}:ro",
        "-v",
        f"{output}:{CONTAINER_OUTPUT_DIR}",
    ]
    if dev:
        command.extend(["-v", f"{MODULE_DIR}:{CONTAINER_MODULE_DIR}:ro"])
    command.extend(
        [
            image,
            "--asset",
            f"{CONTAINER_INPUT_DIR}/{asset.name}",
            "--report",
            f"{CONTAINER_OUTPUT_DIR}/{VALIDATION_REPORT_NAME}",
        ]
    )
    return asset, command
