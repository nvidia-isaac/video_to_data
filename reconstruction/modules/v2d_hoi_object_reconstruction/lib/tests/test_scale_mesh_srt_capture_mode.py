# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from v2d_hoi_object_reconstruction.lib.scale_mesh_srt import estimate_srt_for_frame


def test_two_stage_srt_does_not_silently_use_all_frames(tmp_path):
    with pytest.raises(ValueError, match="Two-stage SRT requires stage1_end_frame"):
        estimate_srt_for_frame(
            job_dir=tmp_path / "job",
            glb_path=tmp_path / "mesh.glb",
            output_dir=tmp_path / "output",
            capture_mode="two_stage",
        )


def test_stationary_srt_rejects_a_stage_boundary(tmp_path):
    with pytest.raises(ValueError, match="incompatible"):
        estimate_srt_for_frame(
            job_dir=tmp_path / "job",
            glb_path=tmp_path / "mesh.glb",
            output_dir=tmp_path / "output",
            capture_mode="stationary",
            stage1_end_frame=100,
        )
