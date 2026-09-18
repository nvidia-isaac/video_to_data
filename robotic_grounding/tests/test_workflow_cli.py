# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise public workflow commands without a Docker daemon or OSMO service."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def commands(tmp_path):
    """Capture the real launchers' external command arguments at the process boundary."""
    command_log = tmp_path / "commands.jsonl"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    for executable in ("docker", "osmo"):
        path = fake_bin / executable
        path.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "with open(os.environ['WORKFLOW_COMMAND_LOG'], 'a') as stream:\n"
            "    stream.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')\n"
            "sys.exit(int(os.environ.get('WORKFLOW_FAKE_EXIT', '0')))\n"
        )
        path.chmod(0o755)
    env = dict(os.environ)
    env.pop("V2D_IMAGE_REGISTRY", None)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["WORKFLOW_COMMAND_LOG"] = str(command_log)

    def run(*args: str, registry=None, failure=None, osmo=False):
        command_env = dict(env)
        if registry is not None:
            command_env["V2D_IMAGE_REGISTRY"] = registry
        if failure is not None:
            command_env["WORKFLOW_FAKE_EXIT"] = str(failure)
        prefix = (
            [
                sys.executable,
                str(ROOT / "scripts/run_osmo.py"),
                "--experiment-name",
                "smoke",
            ]
            if osmo
            else ["bash", str(ROOT / "workflow/run.sh")]
        )
        result = subprocess.run(
            [*prefix, *args],
            env=command_env,
            text=True,
            capture_output=True,
            cwd=tmp_path,
            check=False,
        )
        calls = (
            [json.loads(line) for line in command_log.read_text().splitlines()]
            if command_log.exists()
            else []
        )
        return result, calls

    return run


def test_local_build_does_not_require_registry(commands):
    result, calls = commands("build", "smoke")
    assert result.returncode == 0, result.stderr
    assert calls == [
        [
            "docker",
            "build",
            "-t",
            "robotic-grounding:smoke",
            "-f",
            "workflow/Dockerfile",
            ".",
        ]
    ]


@pytest.mark.parametrize("operation", ["push", "pull", "push-aarch64", "pull-aarch64"])
def test_missing_registry_stops_before_docker(commands, operation):
    result, calls = commands(operation, "smoke")
    assert result.returncode != 0
    assert "V2D_IMAGE_REGISTRY" in result.stdout + result.stderr
    assert calls == []


@pytest.mark.parametrize("suffix", ["", "-aarch64"])
def test_push_uses_configured_namespace(commands, suffix):
    result, calls = commands(
        f"push{suffix}", "smoke", registry="registry.example/team/"
    )
    assert result.returncode == 0, result.stderr
    local = f"robotic-grounding{suffix}:smoke"
    remote = f"registry.example/team/{local}"
    assert calls[:2] == [["docker", "tag", local, remote], ["docker", "push", remote]]


def test_pull_retags_configured_remote_image_for_local_use(commands):
    result, calls = commands("pull", "smoke", registry="registry.example/team/")
    assert result.returncode == 0, result.stderr
    assert calls == [
        ["docker", "pull", "registry.example/team/robotic-grounding:smoke"],
        [
            "docker",
            "tag",
            "registry.example/team/robotic-grounding:smoke",
            "robotic-grounding:smoke",
        ],
    ]


@pytest.mark.parametrize(
    "arguments, error",
    [
        (["--image", "registry.example/team/robotic-grounding:smoke"], "--pool"),
        (["--pool", "my-pool"], "--image"),
        (["--pool", "my-pool", "--build-image"], "V2D_IMAGE_REGISTRY"),
        (
            [
                "--pool",
                "my-pool",
                "--build-image",
                "--image",
                "registry.example/team/custom:v1",
            ],
            "--image",
        ),
    ],
)
def test_submission_requires_explicit_configuration(commands, arguments, error):
    result, calls = commands(*arguments, osmo=True)
    assert result.returncode != 0
    assert error in result.stdout + result.stderr
    assert calls == []


def test_existing_image_submits_without_build_or_registry(commands):
    result, calls = commands(
        "--pool", "my-pool", "--image", "registry.example/team/custom:v1", osmo=True
    )
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    assert calls[0][:3] == ["osmo", "workflow", "submit"]
    assert "image=registry.example/team/custom:v1" in calls[0]


def test_build_push_and_submission_use_same_remote_image(commands):
    result, calls = commands(
        "--pool",
        "my-pool",
        "--build-image",
        registry="registry.example/team/",
        osmo=True,
    )
    assert result.returncode == 0, result.stderr
    assert calls[0][:4] == ["docker", "build", "-t", "robotic-grounding:smoke"]
    assert ["docker", "push", "registry.example/team/robotic-grounding:smoke"] in calls
    assert calls[-1][:3] == ["osmo", "workflow", "submit"]
    assert "image=registry.example/team/robotic-grounding:smoke" in calls[-1]


def test_build_failure_prevents_push_and_submission(commands):
    result, calls = commands(
        "--pool",
        "my-pool",
        "--build-image",
        registry="registry.example/team",
        failure=7,
        osmo=True,
    )
    assert result.returncode != 0
    assert len(calls) == 1
    assert calls[0][:2] == ["docker", "build"]


def test_dry_run_does_not_execute_external_commands(commands):
    result, calls = commands(
        "--pool",
        "my-pool",
        "--build-image",
        "--dry-run",
        registry="registry.example/team/",
        osmo=True,
    )
    assert result.returncode == 0, result.stderr
    assert (
        "image=registry.example/team/robotic-grounding:smoke"
        in result.stdout.replace('"', "")
    )
    assert calls == []
