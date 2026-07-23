# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import inspect
import sys
import types
from pathlib import Path


_RUNNER_PATH = Path(__file__).parents[1] / "run_image_to_mesh.py"


def _load_runner(monkeypatch, captured_calls):
    def fake_run_in_container(**kwargs):
        captured_calls.append(kwargs)

    package_names = ("v2d", "v2d.docker", "v2d.sam3d", "v2d.sam3d.docker")
    for package_name in package_names:
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    container_module = types.ModuleType("v2d.docker.container")
    container_module.run_in_container = fake_run_in_container
    monkeypatch.setitem(sys.modules, "v2d.docker.container", container_module)

    config_module = types.ModuleType("v2d.sam3d.docker._config")
    config_module.IMAGE_NAME = "v2d_sam3d"
    config_module.MODULES_DIR = "/workspace/modules"
    monkeypatch.setitem(sys.modules, "v2d.sam3d.docker._config", config_module)

    spec = importlib.util.spec_from_file_location("sam3d_run_image_to_mesh_test", _RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_arguments():
    return {
        "image_path": "/inputs/image.png",
        "mask_path": "/inputs/mask.png",
        "mesh_path": "/outputs/mesh.glb",
        "transform_path": "/outputs/transform.json",
        "intrinsics_path": "/outputs/intrinsics.json",
        "weights_dir": "/weights/sam3d",
    }


def test_default_preserves_all_gpu_compatibility(monkeypatch):
    captured_calls = []
    module = _load_runner(monkeypatch, captured_calls)

    signature = inspect.signature(module.run_image_to_mesh)
    assert signature.parameters["gpu_device"].default is None

    module.run_image_to_mesh(**_required_arguments())

    assert len(captured_calls) == 1
    assert captured_calls[0]["gpus"] is True
    assert captured_calls[0]["gpu_device"] is None


def test_selected_gpu_is_forwarded_to_container(monkeypatch):
    captured_calls = []
    module = _load_runner(monkeypatch, captured_calls)

    module.run_image_to_mesh(**_required_arguments(), gpu_device=1)

    assert len(captured_calls) == 1
    assert captured_calls[0]["gpus"] is True
    assert captured_calls[0]["gpu_device"] == 1


def test_unprivileged_runtime_caches_are_writable(monkeypatch):
    captured_calls = []
    module = _load_runner(monkeypatch, captured_calls)

    module.run_image_to_mesh(**_required_arguments())

    environment = captured_calls[0]["env"]
    assert environment["MPLCONFIGDIR"].startswith("/tmp/")
    assert environment["WARP_CACHE_PATH"].startswith("/tmp/")
    assert environment["XDG_CACHE_HOME"].startswith("/tmp/")
    assert environment["TRITON_CACHE_DIR"].startswith("/tmp/")


def test_cli_maps_gpu_flag_to_gpu_device():
    source = _RUNNER_PATH.read_text()

    assert 'parser.add_argument("--gpu", type=int, default=None' in source
    assert "gpu_device=args.gpu" in source
