# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for NVBug 6440619 MANO asset staging."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import tempfile
import unittest


_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assets = _load_module("pipeline_mano_assets", "v2d_pipelines/mano_assets.py")


class TestManoAssetLayout(unittest.TestCase):
    def test_hamer_stages_one_canonical_models_asset_and_migrates_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            source = tmp_path / "hand/models/MANO_RIGHT.pkl"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"licensed-mano")
            config = tmp_path / "hamer/_DATA/hamer_ckpts/model_config.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("MANO:\n  MODEL_PATH: data/mano\n")

            staged = assets.prepare_hamer_mano_assets(tmp_path / "hamer", source)

            self.assertEqual(staged, tmp_path / "hamer/_DATA/data/models/MANO_RIGHT.pkl")
            self.assertEqual(staged.read_bytes(), source.read_bytes())
            self.assertIn("MODEL_PATH: data/models", config.read_text())
            self.assertFalse((tmp_path / "hamer/_DATA/data/mano/MANO_RIGHT.pkl").exists())

    def test_wilor_stages_manotorch_models_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            source = tmp_path / "wilor/pretrained_models/MANO_RIGHT.pkl"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"wilor-mano")

            staged = assets.prepare_wilor_manotorch_mano(tmp_path / "wilor")

            self.assertEqual(staged, tmp_path / "wilor/pretrained_models/models/MANO_RIGHT.pkl")
            self.assertEqual(staged.read_bytes(), source.read_bytes())

    def test_wilor_reports_missing_downloaded_mano(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "WiLoR MANO asset"):
                assets.prepare_wilor_manotorch_mano(Path(temp_dir) / "wilor")


if __name__ == "__main__":
    unittest.main()
