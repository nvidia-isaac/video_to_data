#!/usr/bin/env python3
"""OSMO workflow submission script.

SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Script to submit OSMO workflow for robotic grounding development environment.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    args = shlex.split(cmd)
    print(f"Running: {shlex.join(args)}")
    result = subprocess.run(args, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def main() -> None:
    """Submit OSMO workflow for robotic grounding development environment."""
    parser = argparse.ArgumentParser(
        description="Submit OSMO workflow for robotic grounding development"
    )
    parser.add_argument(
        "--experiment-name", required=True, help="Experiment name for the workflow"
    )
    parser.add_argument(
        "--image",
        help="Docker image to use for the workflow (if not provided, will build new one)",
    )
    parser.add_argument(
        "--workflow-yaml",
        default="workflow/train.yaml",
        help="Path to OSMO workflow YAML file",
    )
    parser.add_argument(
        "--pool",
        default="isaac-dev-l40-03",
        help="OSMO pool to use for workflow execution (default: isaac-dev-l40-03)",
    )
    parser.add_argument(
        "--build-image",
        action="store_true",
        help="Build and push Docker image before submitting (uses experiment name as tag).",
    )
    parser.add_argument(
        "--priority",
        default="NORMAL",
        help="OSMO job priority (default: NORMAL)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="extra_sets",
        help="Additional key=value pairs for osmo workflow submit (e.g., dataset=taco)",
    )

    args = parser.parse_args()

    # Get the repository root directory (assuming script is in scripts/)
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)

    # Determine image to use
    if args.build_image:
        # If --image is also specified, build to that exact tag; otherwise tag with experiment name.
        if args.image:
            image_name = args.image
            image_version = image_name.split(":")[-1]
        else:
            image_version = args.experiment_name
            image_name = (
                f"nvcr.io/nvstaging/isaac-amr/robotic-grounding:{image_version}"
            )

        print(f"\nBuilding Docker image: {image_name} ...")
        build_cmd = f"./workflow/run.sh build {image_version}"

        if args.dry_run:
            print(f"[DRY RUN] {build_cmd}")
        else:
            result = run_command(build_cmd, check=False)
            if result.returncode != 0:
                print("Error: Docker build failed")
                sys.exit(1)

        print(f"\nPushing Docker image: {image_name} ...")
        push_cmd = f"./workflow/run.sh push {image_version}"

        if args.dry_run:
            print(f"[DRY RUN] {push_cmd}")
        else:
            result = run_command(push_cmd, check=False)
            if result.returncode != 0:
                print("Error: Docker push failed")
                sys.exit(1)
    elif args.image:
        image_name = args.image
        print(f"Using existing Docker image: {image_name}")
    else:
        image_name = "nvcr.io/nvstaging/isaac-amr/robotic-grounding:latest"
        print(f"Using default Docker image: {image_name}")

    # Use provided image
    print(f"\nUsing Docker image: {image_name}")

    # Check if workflow YAML exists
    workflow_path = Path(args.workflow_yaml)
    if not workflow_path.exists():
        print(f"Error: Workflow file not found: {args.workflow_yaml}")
        sys.exit(1)

    # Submit OSMO workflow
    print(f"\nSubmitting OSMO workflow: {args.workflow_yaml}")
    workflow_name = f"robotic_grounding_{args.experiment_name}"

    all_sets = [
        f'workflow_name="{workflow_name}"',
        f'image="{image_name}"',
    ] + args.extra_sets
    set_str = " ".join(all_sets)

    osmo_cmd = (
        f"osmo workflow submit {args.workflow_yaml} "
        f"--set {set_str} "
        f"--pool {args.pool} "
        f"--priority {args.priority}"
    )

    print(f"\n{osmo_cmd}\n")

    if args.dry_run:
        print("[DRY RUN] Would submit workflow with above command")
    else:
        result = run_command(osmo_cmd, check=False)
        if result.returncode != 0:
            print("Error: OSMO workflow submission failed")
            sys.exit(1)
        print("\n✅ Workflow submitted successfully!")
        print("\nYou can check the workflow status with:")
        print("  osmo workflow list")


if __name__ == "__main__":
    main()
