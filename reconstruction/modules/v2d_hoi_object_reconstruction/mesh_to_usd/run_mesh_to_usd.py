#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert a reconstructed mesh into a rigid USD using Isaac Sim."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from docker_runtime import (
    CONTAINER_INPUT_DIR,
    CONTAINER_MODULE_DIR,
    CONTAINER_OUTPUT_DIR,
    IMAGE_NAME,
    build_isaac_sim_command,
)
from run_validate_usd import run_usd_validation
from validation_docker_runtime import VALIDATION_REPORT_NAME, VALIDATOR_IMAGE_NAME

GENERATION_REPORT_NAME = "generation_report.json"
RIGID_OBJECT_NAME = "rigid_object.usd"
VISUAL_ASSET_NAME = "visual_asset.usd"


def build_mesh_to_usd_command(
    asset_path: str,
    output_dir: str,
    *,
    mass_kg: float | None = None,
    static_friction: float = 0.5,
    dynamic_friction: float = 0.4,
    restitution: float = 0.1,
    max_convex_hulls: int = 16,
    hull_vertex_limit: int = 64,
    coacd_resolution: int = 2000,
    max_decomposition_source_faces: int = 20_000,
    simplify_decomposition_source: bool = True,
    image: str = IMAGE_NAME,
    cache_dir: str | None = None,
    accept_eula: bool = False,
    dev: bool = False,
    gpu_device: str | None = None,
) -> list[str]:
    if mass_kg is not None and mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    if not 0 <= dynamic_friction <= static_friction:
        raise ValueError("Expected 0 <= dynamic_friction <= static_friction")
    if not 0 <= restitution <= 1:
        raise ValueError("restitution must be in [0, 1]")
    if min(
        max_convex_hulls,
        hull_vertex_limit,
        coacd_resolution,
        max_decomposition_source_faces,
    ) <= 0:
        raise ValueError("convex decomposition limits must be positive")

    asset, command = build_isaac_sim_command(
        asset_path,
        output_dir,
        cache_dir=cache_dir,
        accept_eula=accept_eula,
        dev=dev,
        gpu_device=gpu_device,
    )

    runtime = [
        f"{CONTAINER_MODULE_DIR}/runtime/mesh_to_usd.py",
        "--asset", f"{CONTAINER_INPUT_DIR}/{asset.name}",
        "--output-dir", CONTAINER_OUTPUT_DIR,
        "--headless",
        "--mass-kg", str(mass_kg if mass_kg is not None else 0.3),
        "--mass-source", "command_line" if mass_kg is not None else "assumed_grounding_default",
        "--static-friction", str(static_friction),
        "--dynamic-friction", str(dynamic_friction),
        "--restitution", str(restitution),
        "--max-convex-hulls", str(max_convex_hulls),
        "--hull-vertex-limit", str(hull_vertex_limit),
        "--coacd-resolution", str(coacd_resolution),
        "--max-decomposition-source-faces", str(max_decomposition_source_faces),
    ]
    runtime.append(
        "--simplify-decomposition-source"
        if simplify_decomposition_source
        else "--no-simplify-decomposition-source"
    )
    command.extend(["--entrypoint", "/isaac-sim/python.sh", image, *runtime])
    return command


def run_mesh_to_usd(
    asset_path: str,
    output_dir: str,
    *,
    validate: bool = True,
    validator_image: str = VALIDATOR_IMAGE_NAME,
    **kwargs,
) -> dict:
    command = build_mesh_to_usd_command(asset_path, output_dir, **kwargs)
    output_path = Path(output_dir).expanduser().resolve()
    report_path = output_path / GENERATION_REPORT_NAME
    validation_report_path = output_path / VALIDATION_REPORT_NAME
    report_path.unlink(missing_ok=True)
    validation_report_path.unlink(missing_ok=True)

    subprocess.run(command, check=True)
    if not report_path.is_file():
        raise RuntimeError(f"Isaac Sim did not create {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_usd = Path(report.get("output_usd", ""))
    if not output_usd.is_file():
        output_usd = output_path / RIGID_OBJECT_NAME
    visual_asset = Path(report.get("visual_asset", ""))
    if not visual_asset.is_file():
        visual_asset = output_path / VISUAL_ASSET_NAME
    report["output_usd"] = str(output_usd)
    report["visual_asset"] = str(visual_asset)
    if (
        report.get("status") != "generated"
        or not output_usd.is_file()
        or not visual_asset.is_file()
    ):
        raise RuntimeError(
            "Isaac Sim did not produce a complete rigid USD asset package"
        )
    if validate:
        report["simready_validation"] = run_usd_validation(
            str(output_usd),
            output_dir,
            image=validator_image,
            dev=bool(kwargs.get("dev", False)),
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mass-kg", type=float)
    parser.add_argument("--static-friction", type=float, default=0.5)
    parser.add_argument("--dynamic-friction", type=float, default=0.4)
    parser.add_argument("--restitution", type=float, default=0.1)
    parser.add_argument("--max-convex-hulls", type=int, default=16)
    parser.add_argument("--hull-vertex-limit", type=int, default=64)
    parser.add_argument("--coacd-resolution", type=int, default=2000)
    parser.add_argument("--max-decomposition-source-faces", type=int, default=20_000)
    parser.add_argument(
        "--simplify-decomposition-source",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--image", default=IMAGE_NAME)
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validator-image", default=VALIDATOR_IMAGE_NAME)
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--gpu-device",
        help="Expose exactly one GPU to Isaac Sim (for example: 0)",
    )
    parser.add_argument("--accept-eula", action="store_true")
    parser.add_argument("--dev", action="store_true")
    args = vars(parser.parse_args())
    asset = args.pop("asset")
    output_dir = args.pop("output_dir")
    result = run_mesh_to_usd(asset, output_dir, **args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
