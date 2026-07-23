# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import types

import pytest


_DOCKER_DIR = Path(__file__).parents[1]


def _load_runner(monkeypatch, filename: str, captured_calls: list[dict]):
    def fake_run_in_container(**kwargs):
        captured_calls.append(kwargs)

    for package_name in (
        "v2d",
        "v2d.docker",
        "v2d.foundation_pose",
        "v2d.foundation_pose.docker",
    ):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    container_module = types.ModuleType("v2d.docker.container")
    container_module.run_in_container = fake_run_in_container
    monkeypatch.setitem(sys.modules, "v2d.docker.container", container_module)

    config_module = types.ModuleType("v2d.foundation_pose.docker._config")
    config_module.IMAGE_NAME = "v2d_foundation_pose"
    config_module.MODULES_DIR = "/workspace/modules"
    monkeypatch.setitem(
        sys.modules, "v2d.foundation_pose.docker._config", config_module
    )

    runner_path = _DOCKER_DIR / filename
    spec = importlib.util.spec_from_file_location(
        f"foundation_pose_{runner_path.stem}_test", runner_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scale_arguments() -> dict[str, str]:
    return {
        "mesh_path": "/input/mesh.obj",
        "rgb_path": "/input/rgb.png",
        "depth_path": "/input/depth.png",
        "mask_path": "/input/mask.png",
        "intrinsics_path": "/input/intrinsics.json",
        "weights_dir": "/weights/foundation_pose",
        "scale_path": "/output/scale.json",
    }


def _video_arguments() -> dict[str, str]:
    return {
        "video_path": "/input/video.mp4",
        "depth_folder": "/input/depth",
        "masks_folder": "/input/masks",
        "camera_intrinsics_path": "/input/intrinsics.json",
        "mesh_path": "/input/mesh.obj",
        "poses_dir": "/output/poses",
        "weights_dir": "/weights/foundation_pose",
    }


def _ekf_arguments() -> dict[str, str]:
    return {
        "poses_dir": "/input/poses",
        "mesh_path": "/input/mesh.obj",
        "intrinsics_path": "/input/intrinsics.json",
        "weights_dir": "/weights/foundation_pose",
        "output_dir": "/output/poses_smoothed",
    }


@pytest.mark.parametrize(
    ("filename", "function_name", "arguments"),
    [
        (
            "run_estimate_mesh_scale.py",
            "run_estimate_mesh_scale",
            _scale_arguments,
        ),
        ("run_video_to_poses.py", "run_video_to_poses", _video_arguments),
        ("run_ekf_smoothing.py", "run_ekf_smoothing", _ekf_arguments),
    ],
)
def test_default_preserves_legacy_all_gpu_exposure(
    monkeypatch, filename, function_name, arguments
):
    captured_calls: list[dict] = []
    module = _load_runner(monkeypatch, filename, captured_calls)
    function = getattr(module, function_name)

    assert inspect.signature(function).parameters["gpu_device"].default is None
    function(**arguments())

    assert len(captured_calls) == 1
    assert captured_calls[0]["gpus"] is True
    assert captured_calls[0]["gpu_device"] is None
    assert "CUDA_VISIBLE_DEVICES" not in captured_calls[0]["env"]


@pytest.mark.parametrize(
    ("filename", "function_name", "arguments"),
    [
        (
            "run_estimate_mesh_scale.py",
            "run_estimate_mesh_scale",
            _scale_arguments,
        ),
        ("run_video_to_poses.py", "run_video_to_poses", _video_arguments),
        ("run_ekf_smoothing.py", "run_ekf_smoothing", _ekf_arguments),
    ],
)
def test_selected_physical_gpu_is_forwarded_and_remapped_to_cuda_zero(
    monkeypatch, filename, function_name, arguments
):
    captured_calls: list[dict] = []
    module = _load_runner(monkeypatch, filename, captured_calls)
    function = getattr(module, function_name)

    function(**arguments(), gpu_device=1)

    assert len(captured_calls) == 1
    assert captured_calls[0]["gpus"] is True
    assert captured_calls[0]["gpu_device"] == 1
    assert captured_calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.parametrize(
    ("filename", "function_name", "arguments"),
    [
        (
            "run_estimate_mesh_scale.py",
            "run_estimate_mesh_scale",
            _scale_arguments,
        ),
        ("run_video_to_poses.py", "run_video_to_poses", _video_arguments),
        ("run_ekf_smoothing.py", "run_ekf_smoothing", _ekf_arguments),
    ],
)
def test_non_root_runtime_caches_are_writable_tmp_paths(
    monkeypatch, filename, function_name, arguments
):
    captured_calls: list[dict] = []
    module = _load_runner(monkeypatch, filename, captured_calls)

    getattr(module, function_name)(**arguments())

    environment = captured_calls[0]["env"]
    assert environment["MPLCONFIGDIR"] == "/tmp/v2d-matplotlib"
    assert environment["WARP_CACHE_PATH"] == "/tmp/v2d-warp"
    assert environment["XDG_CACHE_HOME"] == "/tmp/v2d-xdg-cache"


@pytest.mark.parametrize(
    ("filename", "function_name", "arguments"),
    [
        (
            "run_estimate_mesh_scale.py",
            "run_estimate_mesh_scale",
            _scale_arguments,
        ),
        ("run_video_to_poses.py", "run_video_to_poses", _video_arguments),
        ("run_ekf_smoothing.py", "run_ekf_smoothing", _ekf_arguments),
    ],
)
@pytest.mark.parametrize("invalid", [-1, True, 1.5])
def test_invalid_gpu_selection_fails_before_container_launch(
    monkeypatch, filename, function_name, arguments, invalid
):
    captured_calls: list[dict] = []
    module = _load_runner(monkeypatch, filename, captured_calls)
    function = getattr(module, function_name)

    with pytest.raises(ValueError, match="non-negative physical GPU index"):
        function(**arguments(), gpu_device=invalid)
    assert captured_calls == []


@pytest.mark.parametrize(
    "filename",
    [
        "run_estimate_mesh_scale.py",
        "run_video_to_poses.py",
        "run_ekf_smoothing.py",
    ],
)
def test_cli_maps_optional_gpu_flag_to_gpu_device(filename: str):
    source = (_DOCKER_DIR / filename).read_text()

    assert '"--gpu"' in source
    assert "default=None" in source
    assert "gpu_device=args.gpu" in source
