# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import trimesh


_PIPELINE_PATH = Path(__file__).parents[1] / "run_ego_wilor.py"


class _StubDependency:
    """Stand-in for pipeline dependencies that this focused test does not call."""


def _load_pipeline(monkeypatch):
    source = _PIPELINE_PATH.read_text()
    imported_modules: dict[str, types.ModuleType] = {}

    for node in ast.parse(source).body:
        if not isinstance(node, ast.ImportFrom) or not node.module.startswith("v2d."):
            continue

        parts = node.module.split(".")
        for end in range(1, len(parts) + 1):
            module_name = ".".join(parts[:end])
            if module_name in imported_modules:
                continue
            module = types.ModuleType(module_name)
            module.__path__ = []
            imported_modules[module_name] = module
            monkeypatch.setitem(sys.modules, module_name, module)
            if end > 1:
                parent_name = ".".join(parts[: end - 1])
                setattr(imported_modules[parent_name], parts[end - 1], module)

        module = imported_modules[node.module]
        for name in node.names:
            setattr(module, name.name, _StubDependency)

    spec = importlib.util.spec_from_file_location("run_ego_wilor_test", _PIPELINE_PATH)
    assert spec is not None and spec.loader is not None
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    return pipeline


def _sorted_vertices(vertices: np.ndarray) -> np.ndarray:
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    return vertices[order]


def test_apply_sam3d_transform_bakes_scene_node_transform_first(
    tmp_path, monkeypatch
):
    pipeline = _load_pipeline(monkeypatch)
    local_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    mesh = trimesh.Trimesh(
        vertices=local_vertices,
        faces=[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        process=False,
    )

    node_transform = trimesh.transformations.rotation_matrix(
        np.pi / 2.0, [1.0, 0.0, 0.0]
    )
    node_transform[:3, 3] = [7.0, -5.0, 11.0]
    scene = trimesh.Scene()
    scene.add_geometry(
        mesh,
        geom_name="object",
        node_name="transformed-object",
        transform=node_transform,
    )
    input_path = tmp_path / "scene.glb"
    input_path.write_bytes(scene.export(file_type="glb"))

    half_sqrt_two = np.sqrt(0.5)
    transform_path = tmp_path / "sam3d_transform.json"
    transform_path.write_text(
        json.dumps(
            {
                "rotation": [half_sqrt_two, 0.0, 0.0, half_sqrt_two],
                "scale": [2.0, 3.0, 4.0],
                "translation": [100.0, 200.0, 300.0],
            }
        )
    )
    output_path = tmp_path / "transformed.ply"

    pipeline._apply_sam3d_transform(
        str(input_path), str(transform_path), str(output_path)
    )

    output = trimesh.load(output_path, force="mesh", process=False)
    scene_vertices = trimesh.transform_points(local_vertices, node_transform)
    sam3d_rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    expected = (
        sam3d_rotation @ np.diag([2.0, 3.0, 4.0]) @ scene_vertices.T
    ).T
    without_scene_transform = (
        sam3d_rotation @ np.diag([2.0, 3.0, 4.0]) @ local_vertices.T
    ).T

    np.testing.assert_allclose(
        _sorted_vertices(output.vertices), _sorted_vertices(expected), atol=1e-6
    )
    assert not np.allclose(
        _sorted_vertices(output.vertices),
        _sorted_vertices(without_scene_transform),
        atol=1e-6,
    )
