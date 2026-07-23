# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def test_inference_selects_and_verifies_the_pinned_moge_snapshot() -> None:
    source = (
        Path(__file__).parents[2] / "lib" / "image_to_mesh.py"
    ).read_text(encoding="utf-8")

    assert "MOGE_REVISION" in source
    assert "MOGE_MODEL_BYTES" in source
    assert "MOGE_MODEL_SHA256" in source
    assert "_verify(Path(moge_model_path)" in source
    assert "os.listdir(cache_path)" not in source
