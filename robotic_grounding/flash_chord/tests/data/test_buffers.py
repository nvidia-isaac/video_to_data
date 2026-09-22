# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for uploading a Reference to GPU Warp buffers."""

import numpy as np
import pytest

from flash_chord.assets import ASSETS_DIR

pytestmark = [pytest.mark.gpu, pytest.mark.sequence_data]

_HOT3D = ASSETS_DIR / "human_motion_data" / "hot3d" / "hot3d_processed" / "sequence_id=P0002_59a84a3a_seg025" / "robot_name=sharpa_wave"


def test_upload_reference_shapes_and_values():
    import warp as wp

    from flash_chord.data.buffers import upload_reference
    from flash_chord.data.mano_sharpa import load_mano_sharpa

    ref = load_mano_sharpa(str(_HOT3D))
    with wp.ScopedDevice("cuda:0"):
        buf = upload_reference(ref)

    assert buf.num_frames == ref.num_frames
    assert buf.sides == ("left", "right")
    assert str(buf.object_body_pos_w.device).startswith("cuda")
    assert buf.object_body_pos_w.shape == (ref.num_frames, 2, 3)  # hot3d: 2 object bodies
    for s in buf.sides:
        assert buf.wrist_pos_w[s].shape == (ref.num_frames, 3)
        assert buf.wrist_quat_w[s].shape == (ref.num_frames, 4)
    # frame-0 round-trips back to the host reference
    np.testing.assert_allclose(buf.wrist_pos_w["left"].numpy()[0], ref.wrist_pos_w("left")[0], atol=1e-5)
    np.testing.assert_allclose(buf.object_body_pos_w.numpy()[0], ref.object_body_pos_w()[0], atol=1e-5)
