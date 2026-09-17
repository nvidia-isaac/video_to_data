# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Container-only contracts; invoke with python -m v2d.cari4d.docker.run_tests."""

import ast
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from ._contract_helpers import MODULE, _function


def test_side_by_side_frame_preserves_rgb_and_uses_uniform_background():
    renderer_path = MODULE / "lib/cari4d/tools/render_mhr_wild_inference.py"
    function = _function(renderer_path, "_side_by_side_frame")
    namespace = {"np": np, "MESH_BACKGROUND": np.array([32, 36, 40], dtype=np.uint8)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(renderer_path), "exec"), namespace)
    rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    rendered = np.zeros((2, 4, 3), dtype=np.float32)
    rendered[0, 1] = [1.0, 0.5, 0.0]
    foreground = np.zeros((2, 4), dtype=bool)
    foreground[0, 1] = True
    frame = namespace["_side_by_side_frame"](rgb, rendered, foreground)
    assert frame.shape == (2, 8, 3)
    assert np.array_equal(frame[:, :4], rgb)
    assert np.array_equal(frame[0, 5], [255, 128, 0])
    assert np.all(frame[:, 4:][~foreground] == [32, 36, 40])


def test_three_stage_overlay_blends_render_over_rgb():
    renderer_path = MODULE / "lib/cari4d/tools/render_mhr_wild_inference.py"
    function = _function(renderer_path, "_overlay_frame")
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(renderer_path), "exec"), namespace)
    rgb = np.full((2, 3, 3), 100, dtype=np.uint8)
    rendered = np.zeros((2, 3, 3), dtype=np.float32)
    rendered[0, 1] = [1.0, 0.0, 0.0]
    foreground = np.zeros((2, 3), dtype=bool)
    foreground[0, 1] = True
    frame = namespace["_overlay_frame"](rgb, rendered, foreground)
    assert frame.shape == rgb.shape
    assert np.array_equal(frame[1, 1], [100, 100, 100])
    assert np.array_equal(frame[0, 1], [212, 28, 28])


def test_three_stage_comparison_has_expected_panel_and_header_dimensions():
    import cv2

    renderer_path = MODULE / "lib/cari4d/tools/render_mhr_wild_inference.py"
    functions = {node.name: node for node in ast.parse(renderer_path.read_text()).body if isinstance(node, ast.FunctionDef)}
    namespace = {"cv2": cv2, "np": np}
    for name in ("_overlay_frame", "_header", "_comparison_frame"):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[functions[name]], type_ignores=[])), str(renderer_path), "exec"), namespace)
    rgb = np.full((20, 30, 3), 100, dtype=np.uint8)
    rendered = np.zeros_like(rgb, dtype=np.float32)
    foreground = np.zeros(rgb.shape[:2], dtype=bool)
    frame = namespace["_comparison_frame"](rgb, rendered, foreground, rendered, foreground, rendered, foreground, "000123", "Contact-guided refinement")
    assert frame.shape == (104, 90, 3)


def test_sam2_mask_packer_writes_exact_cari4d_keys(tmp_path):
    packer_path = MODULE / "lib/pack_masks.py"
    functions = {node.name: node for node in ast.parse(packer_path.read_text()).body if isinstance(node, ast.FunctionDef)}
    video_path = tmp_path / "example.0.color.mp4"
    human_path = tmp_path / "human"
    object_path = tmp_path / "object"
    output_path = tmp_path / "example_masks_k0.h5"
    video_path.write_bytes(b"video")
    human_path.mkdir()
    object_path.mkdir()

    class FakeSource:
        def __init__(self, frames, image_size, stems):
            self.frames = frames
            self.n_frames = len(frames)
            self.image_size = image_size
            self.stems = stems

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def iter_frames(self):
            yield from self.frames

    frames = [np.zeros((2, 3), dtype=np.uint8), np.ones((2, 3), dtype=np.uint8)]
    sources = {video_path.resolve(): FakeSource([None, None], (3, 2), ["000000", "000001"]), human_path.resolve(): FakeSource(frames, (3, 2), ["000000", "000001"]), object_path.resolve(): FakeSource(frames[::-1], (3, 2), ["000000", "000001"])}

    class FakeFrameSource:
        @classmethod
        def from_path(cls, path):
            return sources[Path(path).resolve()]

    namespace = {"Path": Path, "FrameSource": FakeFrameSource, "h5py": h5py, "np": np, "os": os, "MASK_SCHEMA": "v2d.cari4d.wild_masks.v1"}
    for name in ("_sequence_name", "_mask", "masks_pack_cari4d_h5"):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[functions[name]], type_ignores=[])), str(packer_path), "exec"), namespace)
    assert namespace["masks_pack_cari4d_h5"](video_path, human_path, object_path, output_path) == output_path.resolve()
    with h5py.File(output_path, "r") as output:
        assert output.attrs["schema"] == "v2d.cari4d.wild_masks.v1"
        assert output.attrs["frame_count"] == 2
        assert set(output["example"]) == {"000000-k0.person_mask.png", "000000-k0.obj_rend_mask.png", "000001-k0.person_mask.png", "000001-k0.obj_rend_mask.png"}
        assert np.all(output["example/000001-k0.person_mask.png"][:] == 255)


def test_wild_device_transfer_preserves_non_tensor_batch_metadata():
    path = MODULE / "lib/cari4d/tools/run_mhr_wild_inference.py"
    function = _function(path, "_to_device")
    namespace = {"Any": Any, "Mapping": Mapping, "torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"), namespace)
    tensor = torch.tensor([1.0])
    batch = namespace["_to_device"]({"input_xyz": tensor, "_mhr_spatial_normalization_applied": True}, torch.device("cpu"))
    assert torch.equal(batch["input_xyz"], tensor)
    assert batch["_mhr_spatial_normalization_applied"] is True
