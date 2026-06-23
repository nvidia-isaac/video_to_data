#!/usr/bin/env python3
"""
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

Script to submit OSMO workflows for the Video Ingestion Agent.

Supports three workflow types:
  - benchmark  (default): EPIC-KITCHENS benchmark evaluation
  - batch_ingestion:      Large-scale video ingestion into entity graph DB
  - webapp:               Host the Gradio web interface with vLLM server
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


# ---------------------------------------------------------------------------
# Workflow-specific --set builders
# ---------------------------------------------------------------------------

WORKFLOW_DEFAULTS = {
    "benchmark": "osmo_workflows/benchmark.yaml",
    "batch_ingestion": "osmo_workflows/batch_ingestion.yaml",
    "webapp": "osmo_workflows/webapp.yaml",
}


def _build_benchmark_sets(args, image_name: str, hf_token: str, nim_api_key: str) -> list[str]:
    """Build --set key=value pairs for the benchmark workflow."""
    wandb_api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY", "")
    workflow_name = f"benchmark_{args.experiment_name}"
    return [
        f'workflow_name="{workflow_name}"',
        f'image="{image_name}"',
        f'hf_token="{hf_token}"',
        f'nim_api_key="{nim_api_key}"',
        f'wandb_project="{args.wandb_project}"',
        f'wandb_run_name="{args.experiment_name}"',
        f'wandb_api_key="{wandb_api_key}"',
    ]


def _build_batch_ingestion_sets(
    args, image_name: str, hf_token: str, nim_api_key: str
) -> list[str]:
    """Build --set key=value pairs for the batch ingestion workflow."""
    workflow_name = f"ingest_{args.experiment_name}"
    sets = [
        f'workflow_name="{workflow_name}"',
        f'image="{image_name}"',
        f'hf_token="{hf_token}"',
        f'nim_api_key="{nim_api_key}"',
        f'experiment_name="{args.experiment_name}"',
    ]
    if args.output_base_dir:
        sets.append(f'output_base_dir="{args.output_base_dir}"')
    if args.num_shards:
        sets.append(f'num_shards="{args.num_shards}"')
    if args.input_dir:
        sets.append(f'input_dir="{args.input_dir}"')
    return sets


def _build_webapp_sets(args, image_name: str, hf_token: str, nim_api_key: str) -> list[str]:
    """Build --set key=value pairs for the webapp workflow."""
    workflow_name = f"webapp_{args.experiment_name}"
    sets = [
        f'workflow_name="{workflow_name}"',
        f'image="{image_name}"',
        f'hf_token="{hf_token}"',
        f'nim_api_key="{nim_api_key}"',
    ]
    if args.nfs_db_dir:
        sets.append(f'nfs_db_dir="{args.nfs_db_dir}"')
    if args.webapp_port:
        sets.append(f'webapp_port="{args.webapp_port}"')
    return sets


def main():
    parser = argparse.ArgumentParser(
        description="Submit OSMO workflow for the Video Ingestion Agent (benchmark, batch ingestion, or webapp)"
    )
    parser.add_argument(
        "workflow_type",
        nargs="?",
        default="benchmark",
        choices=["benchmark", "batch_ingestion", "webapp"],
        help="Workflow type to submit (default: benchmark)",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment name (used for image tag, workflow name, and output subdirectory)",
    )
    parser.add_argument(
        "--image",
        help="Docker image to use (if not provided, will build and push a new one)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token (defaults to HF_TOKEN environment variable)",
    )
    parser.add_argument(
        "--nim-api-key",
        default=None,
        help="NVIDIA NIM API key for query agent (defaults to NIM_API_KEY env var)",
    )
    parser.add_argument(
        "--workflow-yaml",
        default=None,
        help="Path to OSMO workflow YAML file (auto-detected from workflow_type if omitted)",
    )
    parser.add_argument(
        "--pool",
        default="isaac-dev-h100-01",
        help="OSMO pool to use for workflow execution (default: isaac-dev-h100-01)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    # Benchmark-specific options
    bench_group = parser.add_argument_group("benchmark options")
    bench_group.add_argument(
        "--wandb-api-key",
        default=None,
        help="Weights & Biases API key (defaults to WANDB_API_KEY env var)",
    )
    bench_group.add_argument(
        "--wandb-project",
        default="v2p-benchmark",
        help="W&B project name (default: v2p-benchmark)",
    )

    # Batch-ingestion-specific options
    batch_group = parser.add_argument_group("batch ingestion options")
    batch_group.add_argument(
        "--output-base-dir",
        default=None,
        help="NFS base directory for DB outputs (default from workflow YAML)",
    )
    batch_group.add_argument(
        "--num-shards",
        default=None,
        help="Number of parallel shards (default from workflow YAML)",
    )
    batch_group.add_argument(
        "--input-dir",
        default=None,
        help="Override input_dir inside the container",
    )

    # Webapp-specific options
    webapp_group = parser.add_argument_group("webapp options")
    webapp_group.add_argument(
        "--nfs-db-dir",
        default=None,
        help="NFS path to entity graph DB directory (default from workflow YAML)",
    )
    webapp_group.add_argument(
        "--webapp-port",
        default=None,
        help="Port for the Gradio webapp (default: 7860)",
    )

    args = parser.parse_args()

    # Resolve workflow YAML
    workflow_yaml = args.workflow_yaml or WORKFLOW_DEFAULTS[args.workflow_type]

    # Get HF token from args or environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        print(
            "Error: HF_TOKEN is required. Provide via --hf-token or HF_TOKEN environment variable"
        )
        sys.exit(1)

    # Get NIM API key from args or environment
    nim_api_key = args.nim_api_key or os.environ.get("NIM_API_KEY", "")

    # Build or use existing image
    if args.image:
        image_name = args.image
        print(f"Using existing Docker image: {image_name}")
    else:
        # Generate image name with latest tag
        image_name = (
            f"nvcr.io/nvstaging/isaac-amr/v2p_{args.workflow_type}_{args.experiment_name}:latest"
        )

        print(f"\nBuilding Docker image: {image_name}")
        build_cmd = f"docker build --network=host -t {image_name} -f Dockerfile ."

        if args.dry_run:
            print(f"[DRY RUN] {build_cmd}")
        else:
            result = run_command(build_cmd, check=False)
            if result.returncode != 0:
                print("Error: Docker build failed")
                sys.exit(1)

        print(f"\nPushing Docker image to NGC: {image_name}")
        push_cmd = f"docker push {image_name}"

        if args.dry_run:
            print(f"[DRY RUN] {push_cmd}")
        else:
            result = run_command(push_cmd, check=False)
            if result.returncode != 0:
                print("Error: Docker push failed")
                sys.exit(1)

    print(f"\nUsing Docker image: {image_name}")

    # Check if workflow YAML exists
    workflow_path = Path(workflow_yaml)
    if not workflow_path.exists():
        print(f"Error: Workflow file not found: {workflow_yaml}")
        sys.exit(1)

    # Build --set pairs for the selected workflow type
    if args.workflow_type == "benchmark":
        set_pairs = _build_benchmark_sets(args, image_name, hf_token, nim_api_key)
    elif args.workflow_type == "batch_ingestion":
        set_pairs = _build_batch_ingestion_sets(args, image_name, hf_token, nim_api_key)
    else:
        set_pairs = _build_webapp_sets(args, image_name, hf_token, nim_api_key)

    # Assemble osmo command
    print(f"\nSubmitting OSMO workflow: {workflow_yaml}")

    sets_str = " ".join(set_pairs)
    osmo_cmd = f"osmo workflow submit {workflow_yaml} --set {sets_str} --pool {args.pool}"

    print(f"\n{osmo_cmd}\n")

    if args.dry_run:
        print("[DRY RUN] Would submit workflow with above command")
    else:
        result = run_command(osmo_cmd, check=False)
        if result.returncode != 0:
            print("Error: OSMO workflow submission failed")
            sys.exit(1)
        print("\nWorkflow submitted successfully!")


if __name__ == "__main__":
    main()
