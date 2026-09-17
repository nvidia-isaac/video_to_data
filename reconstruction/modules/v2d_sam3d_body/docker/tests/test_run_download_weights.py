# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
import types
from pathlib import Path


_DOCKER_DIR = Path(__file__).parents[1]


def _load_module(monkeypatch):
    config_name = "v2d.sam3d_body.docker._config"
    config = types.ModuleType(config_name)
    config.IMAGE_NAME = "v2d_sam3d_body"
    config.MODULES_DIR = "/workspace/modules"
    monkeypatch.setitem(sys.modules, config_name, config)

    spec = importlib.util.spec_from_file_location(
        "test_sam3d_body_run_download_weights",
        _DOCKER_DIR / "run_download_weights.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _capture_run(monkeypatch, module):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return captured


def _assert_token_forwarded_without_argv_exposure(captured, secret):
    cmd = captured["cmd"]
    assert secret not in "\0".join(cmd)
    assert any(
        cmd[index : index + 2] == ["-e", "HF_TOKEN"]
        for index in range(len(cmd) - 1)
    )
    assert captured["kwargs"]["env"]["HF_TOKEN"] == secret
    assert captured["kwargs"]["check"] is True


def test_environment_token_is_forwarded_without_argv_exposure(
    monkeypatch, tmp_path,
):
    secret = "hf_environment_secret"
    monkeypatch.setenv("HF_TOKEN", secret)
    module = _load_module(monkeypatch)
    captured = _capture_run(monkeypatch, module)
    output_dir = tmp_path / "weights"

    module.run_download(str(output_dir))

    _assert_token_forwarded_without_argv_exposure(captured, secret)
    cmd = captured["cmd"]
    assert cmd[:7] == [
        "docker", "run", "--rm", "--gpus", "all", "--user",
        f"{os.getuid()}:{os.getgid()}",
    ]
    assert ["-e", "HF_HOME=/tmp/hf_cache"] == cmd[7:9]
    assert f"{output_dir.resolve()}:/data/weights" in cmd
    assert cmd[-6:] == [
        "v2d_sam3d_body", "python", "-m",
        "v2d.sam3d_body.lib.download_weights", "--output_dir",
        "/data/weights",
    ]


def test_cached_token_is_forwarded_without_argv_exposure(
    monkeypatch, tmp_path,
):
    secret = "hf_cached_secret"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    token_path = tmp_path / ".cache" / "huggingface" / "token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text(secret)
    module = _load_module(monkeypatch)
    captured = _capture_run(monkeypatch, module)

    module.run_download(str(tmp_path / "weights"), dev=True)

    _assert_token_forwarded_without_argv_exposure(captured, secret)
    assert ["-v", "/workspace/modules:/workspace"] == captured["cmd"][-8:-6]


def test_missing_token_does_not_add_hf_token_option(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    module = _load_module(monkeypatch)
    captured = _capture_run(monkeypatch, module)

    module.run_download(str(tmp_path / "weights"))

    assert "HF_TOKEN" not in captured["cmd"]
    assert "HF_TOKEN" not in captured["kwargs"]["env"]
