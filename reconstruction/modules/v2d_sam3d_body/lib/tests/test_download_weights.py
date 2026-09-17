# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).parents[1] / "download_weights.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "test_sam3d_body_download_weights", _MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_artifacts(directory, relative_paths):
    for relative_path in relative_paths:
        path = directory / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")


def _write_all_artifacts(module, output_dir):
    _write_artifacts(
        output_dir / "sam-3d-body-dinov3",
        module.SAM3D_REQUIRED_ARTIFACTS,
    )
    _write_artifacts(
        output_dir / "moge-2-vitb-normal",
        module.MOGE_REQUIRED_ARTIFACTS,
    )
    _write_artifacts(
        output_dir / module.DINOV3_CACHE_DIR,
        module.DINOV3_REQUIRED_ARTIFACTS,
    )


def _install_successful_runner(monkeypatch, module, calls):
    repository_artifacts = {
        "facebook/sam-3d-body-dinov3": module.SAM3D_REQUIRED_ARTIFACTS,
        "Ruicheng/moge-2-vitb-normal": module.MOGE_REQUIRED_ARTIFACTS,
    }

    def run(command, check):
        assert check is True
        calls.append(command)
        if command[:2] == ["hf", "download"]:
            destination = Path(command[command.index("--local-dir") + 1])
            _write_artifacts(destination, repository_artifacts[command[2]])
        else:
            assert command[:2] == ["git", "clone"]
            _write_artifacts(Path(command[-1]), module.DINOV3_REQUIRED_ARTIFACTS)

    monkeypatch.setattr(module.subprocess, "run", run)


def test_complete_artifacts_skip_downloads_and_authentication(monkeypatch, tmp_path):
    module = _load_module()
    _write_all_artifacts(module, tmp_path)
    monkeypatch.setattr(
        module,
        "_ensure_hf_token",
        lambda: pytest.fail("authentication should not be checked"),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("download should not run"),
    )

    module.download_weights(str(tmp_path))


def test_empty_output_downloads_and_validates_every_dependency(monkeypatch, tmp_path):
    module = _load_module()
    calls = []
    auth_checks = []
    monkeypatch.setattr(module, "_ensure_hf_token", lambda: auth_checks.append(True))
    _install_successful_runner(monkeypatch, module, calls)

    module.download_weights(str(tmp_path))

    assert auth_checks == [True]
    assert [command[:2] for command in calls] == [
        ["hf", "download"],
        ["hf", "download"],
        ["git", "clone"],
    ]
    assert not module._missing_artifacts(
        tmp_path / "sam-3d-body-dinov3", module.SAM3D_REQUIRED_ARTIFACTS,
    )
    assert not module._missing_artifacts(
        tmp_path / "moge-2-vitb-normal", module.MOGE_REQUIRED_ARTIFACTS,
    )
    assert not module._missing_artifacts(
        tmp_path / module.DINOV3_CACHE_DIR, module.DINOV3_REQUIRED_ARTIFACTS,
    )


def test_partial_hugging_face_downloads_resume_in_existing_directories(
    monkeypatch, tmp_path,
):
    module = _load_module()
    sam3d_dir = tmp_path / "sam-3d-body-dinov3"
    moge_dir = tmp_path / "moge-2-vitb-normal"
    _write_artifacts(sam3d_dir, ("LICENSE", "model.ckpt"))
    _write_artifacts(moge_dir, ("README.md",))
    (sam3d_dir / "model_config.yaml").touch()
    (moge_dir / "model.pt").touch()
    _write_artifacts(
        tmp_path / module.DINOV3_CACHE_DIR,
        module.DINOV3_REQUIRED_ARTIFACTS,
    )
    calls = []
    monkeypatch.setattr(module, "_ensure_hf_token", lambda: None)
    _install_successful_runner(monkeypatch, module, calls)

    module.download_weights(str(tmp_path))

    assert [Path(command[-1]) for command in calls] == [sam3d_dir, moge_dir]
    assert (sam3d_dir / "LICENSE").is_file()
    assert (moge_dir / "README.md").is_file()


def test_interrupted_hugging_face_download_is_retried(monkeypatch, tmp_path):
    module = _load_module()
    _write_artifacts(
        tmp_path / "moge-2-vitb-normal", module.MOGE_REQUIRED_ARTIFACTS,
    )
    _write_artifacts(
        tmp_path / module.DINOV3_CACHE_DIR,
        module.DINOV3_REQUIRED_ARTIFACTS,
    )
    sam3d_dir = tmp_path / "sam-3d-body-dinov3"
    _write_artifacts(sam3d_dir, ("LICENSE",))
    monkeypatch.setattr(module, "_ensure_hf_token", lambda: None)

    def fail_download(command, check):
        _write_artifacts(sam3d_dir, ("model.ckpt",))
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module.subprocess, "run", fail_download)
    with pytest.raises(subprocess.CalledProcessError):
        module.download_weights(str(tmp_path))

    calls = []
    _install_successful_runner(monkeypatch, module, calls)
    module.download_weights(str(tmp_path))

    assert len(calls) == 1
    assert Path(calls[0][-1]) == sam3d_dir
    assert (sam3d_dir / "model.ckpt").is_file()


@pytest.mark.parametrize(
    ("component", "missing_artifact"),
    [
        ("sam3d", "assets/mhr_model.pt"),
        ("moge", "model.pt"),
        ("dinov3", "dinov3/__init__.py"),
    ],
)
def test_successful_command_without_required_artifact_fails(
    monkeypatch, tmp_path, component, missing_artifact,
):
    module = _load_module()
    _write_all_artifacts(module, tmp_path)
    component_dirs = {
        "sam3d": tmp_path / "sam-3d-body-dinov3",
        "moge": tmp_path / "moge-2-vitb-normal",
        "dinov3": tmp_path / module.DINOV3_CACHE_DIR,
    }
    (component_dirs[component] / missing_artifact).unlink()
    monkeypatch.setattr(module, "_ensure_hf_token", lambda: None)

    def incomplete_download(command, check):
        if command[:2] == ["git", "clone"]:
            _write_artifacts(Path(command[-1]), ("hubconf.py",))

    monkeypatch.setattr(module.subprocess, "run", incomplete_download)

    with pytest.raises(RuntimeError, match=missing_artifact):
        module.download_weights(str(tmp_path))


def test_dinov3_replaced_only_after_valid_clone(monkeypatch, tmp_path):
    module = _load_module()
    _write_artifacts(
        tmp_path / "sam-3d-body-dinov3", module.SAM3D_REQUIRED_ARTIFACTS,
    )
    _write_artifacts(
        tmp_path / "moge-2-vitb-normal", module.MOGE_REQUIRED_ARTIFACTS,
    )
    dinov3_dir = tmp_path / module.DINOV3_CACHE_DIR
    _write_artifacts(dinov3_dir, ("partial.txt",))
    monkeypatch.setattr(module, "_ensure_hf_token", lambda: None)

    def incomplete_clone(command, check):
        _write_artifacts(Path(command[-1]), ("hubconf.py",))

    monkeypatch.setattr(module.subprocess, "run", incomplete_clone)
    with pytest.raises(RuntimeError):
        module.download_weights(str(tmp_path))
    assert (dinov3_dir / "partial.txt").is_file()

    calls = []
    _install_successful_runner(monkeypatch, module, calls)
    module.download_weights(str(tmp_path))

    assert not (dinov3_dir / "partial.txt").exists()
    assert not module._missing_artifacts(
        dinov3_dir, module.DINOV3_REQUIRED_ARTIFACTS,
    )
