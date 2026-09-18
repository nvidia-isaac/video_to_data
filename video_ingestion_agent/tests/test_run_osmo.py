# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Exercise the public OSMO CLI without building images or submitting jobs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def run_osmo(*args, repository=None):
    env = os.environ.copy()
    env.pop("V2D_IMAGE_REPOSITORY", None)
    env.pop("NIM_API_KEY", None)
    env.pop("WANDB_API_KEY", None)
    env["HF_TOKEN"] = "test-token"
    if repository is not None:
        env["V2D_IMAGE_REPOSITORY"] = repository
    return subprocess.run(
        [sys.executable, "scripts/run_osmo.py", *args, "--experiment-name", "smoke", "--dry-run"],
        cwd=PACKAGE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_existing_image_does_not_select_an_internal_pool():
    result = run_osmo("webapp", "--image", "ghcr.io/example/agent:v1")
    assert result.returncode == 0, result.stderr
    assert 'image="ghcr.io/example/agent:v1"' in result.stdout
    assert "--pool" not in result.stdout
    assert "docker build" not in result.stdout


@pytest.mark.parametrize("use_env", [False, True])
def test_build_uses_configured_image_repository(use_env):
    args = [] if use_env else ["--image-repository", "ghcr.io/example/"]
    result = run_osmo("webapp", *args, repository="ghcr.io/example/" if use_env else None)
    assert result.returncode == 0, result.stderr
    assert "docker build --network=host -t ghcr.io/example/v2p_webapp_smoke:latest" in result.stdout
    assert "docker push ghcr.io/example/v2p_webapp_smoke:latest" in result.stdout


def test_missing_image_configuration_fails_before_build():
    result = run_osmo("webapp")
    assert result.returncode != 0
    assert "--image" in result.stderr
    assert "docker build" not in result.stdout


def test_cli_repository_overrides_environment():
    result = run_osmo(
        "webapp", "--image-repository", "ghcr.io/selected", repository="ghcr.io/environment"
    )
    assert result.returncode == 0, result.stderr
    assert "ghcr.io/selected/v2p_webapp_smoke:latest" in result.stdout
    assert "ghcr.io/environment" not in result.stdout


@pytest.mark.parametrize("workflow", ["benchmark", "batch_ingestion"])
def test_video_workflows_require_explicit_source(workflow):
    result = run_osmo(workflow, "--image", "ghcr.io/example/agent:v1")
    assert result.returncode != 0
    assert "--nfs-videos-path" in result.stderr


def test_batch_passes_configured_paths_and_pool():
    result = run_osmo(
        "batch_ingestion",
        "--image",
        "ghcr.io/example/agent:v1",
        "--pool",
        "my-gpu-pool",
        "--nfs-videos-path",
        "/mnt/videos",
        "--output-base-dir",
        "/mnt/results",
    )
    assert result.returncode == 0, result.stderr
    assert 'nfs_videos_path="/mnt/videos"' in result.stdout
    assert 'output_base_dir="/mnt/results"' in result.stdout
    assert "--pool my-gpu-pool" in result.stdout


def test_batch_requires_output_destination():
    result = run_osmo(
        "batch_ingestion",
        "--image",
        "ghcr.io/example/agent:v1",
        "--nfs-videos-path",
        "/mnt/videos",
    )
    assert result.returncode != 0
    assert "--output-base-dir" in result.stderr


@pytest.mark.parametrize("option", ["--nfs-videos-path", "--output-base-dir", "--nfs-db-dir"])
def test_relative_cluster_paths_fail_before_submission(option):
    result = run_osmo(
        "batch_ingestion",
        "--image",
        "ghcr.io/example/agent:v1",
        "--nfs-videos-path",
        "/mnt/videos",
        "--output-base-dir",
        "/mnt/results",
        option,
        "relative/path",
    )
    assert result.returncode != 0
    assert f"{option} must be an absolute path" in result.stderr
    assert "osmo workflow submit" not in result.stdout


def test_benchmark_passes_configured_source():
    result = run_osmo(
        "benchmark",
        "--image",
        "ghcr.io/example/agent:v1",
        "--nfs-videos-path",
        "/mnt/epic-kitchens",
    )
    assert result.returncode == 0, result.stderr
    assert 'nfs_videos_path="/mnt/epic-kitchens"' in result.stdout


def test_webapp_passes_database_path_to_template_parameter():
    result = run_osmo(
        "webapp",
        "--image",
        "ghcr.io/example/agent:v1",
        "--nfs-db-dir",
        "/mnt/database",
    )
    assert result.returncode == 0, result.stderr
    assert 'default_db_dir="/mnt/database"' in result.stdout


def test_missing_workflow_fails_before_build():
    result = run_osmo(
        "webapp",
        "--image-repository",
        "ghcr.io/example",
        "--workflow-yaml",
        "missing.yaml",
    )
    assert result.returncode != 0
    assert "Workflow file not found" in result.stderr
    assert "docker build" not in result.stdout


def test_placeholder_image_is_rejected():
    result = run_osmo(
        "webapp", "--image", "image-registry-not-configured-see-readme.invalid/agent:v1"
    )
    assert result.returncode != 0
    assert "--image" in result.stderr
