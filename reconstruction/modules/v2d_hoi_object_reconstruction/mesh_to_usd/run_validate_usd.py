#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a generated rigid USD with the scoped V2D SimReady profile."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from validation_docker_runtime import (
    VALIDATION_REPORT_NAME,
    VALIDATOR_IMAGE_NAME,
    build_validator_command,
)


def build_validation_command(
    asset_path: str,
    output_dir: str,
    *,
    image: str = VALIDATOR_IMAGE_NAME,
    dev: bool = False,
) -> list[str]:
    _asset, command = build_validator_command(
        asset_path,
        output_dir,
        image=image,
        dev=dev,
    )
    return command


def run_usd_validation(
    asset_path: str,
    output_dir: str,
    *,
    image: str = VALIDATOR_IMAGE_NAME,
    dev: bool = False,
) -> dict:
    asset = Path(asset_path).expanduser().resolve()
    report_path = Path(output_dir).expanduser().resolve() / VALIDATION_REPORT_NAME
    command = build_validation_command(
        str(asset),
        output_dir,
        image=image,
        dev=dev,
    )
    report_path.unlink(missing_ok=True)
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if not report_path.is_file():
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"USD validator exited with code {completed.returncode} without creating "
            f"{report_path}"
            + (f": {detail}" if detail else "")
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    container_asset = report.get("asset")
    if container_asset and container_asset != str(asset):
        report["container_asset"] = container_asset
    report["asset"] = str(asset)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report.get("status") == "error":
        raise RuntimeError(
            f"USD validation could not run: {report.get('error', 'unknown error')}"
        )
    if not report.get("passed", False):
        failed_requirements = report.get("failed_requirements") or [
            "unspecified requirement"
        ]
        detail = ", ".join(str(value) for value in failed_requirements)
        raise RuntimeError(
            f"USD validation failed: {detail}. Review {report_path}"
        )
    completed.check_returncode()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image", default=VALIDATOR_IMAGE_NAME)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    result = run_usd_validation(
        args.asset,
        args.output_dir,
        image=args.image,
        dev=args.dev,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
