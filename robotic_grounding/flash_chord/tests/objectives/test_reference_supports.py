# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Demo (command) wrench-support precompute over a real reference."""

import numpy as np
import pytest

from flash_chord.assets import ASSETS_DIR

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.sequence_data]

_BOX = ASSETS_DIR / "human_motion_data" / "arctic" / "arctic_processed" / "sequence_id=dataset_s07_box_grab_01" / "robot_name=sharpa_wave"


def test_reference_supports_shape_and_activity():
    import warp as wp

    from flash_chord.data.mano_sharpa import load_mano_sharpa
    from flash_chord.objectives.wrench import compute_reference_supports

    ref = load_mano_sharpa(str(_BOX))
    num_basis = 64
    with wp.ScopedDevice("cuda:0"):
        sup = compute_reference_supports(ref, num_basis=num_basis, num_edges=8, mu=0.1, device="cuda:0")

    T, B = ref.object_body_pos_w().shape[0], ref.object_body_pos_w().shape[1]
    assert sup.shape == (T, len(ref.sides), B, num_basis)
    assert np.isfinite(sup).all() and (sup >= 0.0).all()  # support is clamped non-negative
    assert sup.max() > 1e-3  # the demo grasp makes contact, so some directions are supported
