# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the contact-wrench objective / penalty kernels (support, missed, unintended)."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _launch(kernel, cmd, cur, n_hands, n_bodies, n_basis, extra=()):
    import warp as wp

    with wp.ScopedDevice("cuda:0"):
        c = wp.array(np.asarray(cmd, dtype=np.float32), dtype=wp.float32)
        u = wp.array(np.asarray(cur, dtype=np.float32), dtype=wp.float32)
        out = wp.zeros(1, dtype=wp.float32)
        wp.launch(kernel, dim=1, inputs=[c, u, n_hands, n_bodies, n_basis, *extra], outputs=[out])
        return float(out.numpy()[0])


def test_support_objective_tolerance_band():
    from flash_chord.objectives.contact import contact_wrench_support_objective

    # 1 world/hand/body, 4 basis dirs; 2 commanded-active (dirs 0,1). dir0 matches; dir1 under-supported.
    cmd = [1.0, 1.0, 0.0, 0.0]
    cur = [1.0, 0.5, 0.0, 0.0]
    tol, var = 0.1, 0.1
    r = _launch(contact_wrench_support_objective, cmd, cur, 1, 1, 4, extra=(tol, var))
    # dir0: better/too_large both 0 -> exp(0)=1; dir1: better=0.9-0.5=0.4 -> loss 0.16 -> exp(-1.6); /cmd_num=2
    expected = (1.0 + np.exp(-0.16 / var)) / 2.0
    assert abs(r - expected) < 1e-5


def test_missed_penalty_fraction():
    from flash_chord.objectives.contact import missed_contact_penalty

    # 3 commanded dirs (0,1,2); agent covers 0 and 2, misses 1 -> missing fraction 1/3.
    r = _launch(missed_contact_penalty, [1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0], 1, 1, 4)
    assert abs(r - 1.0 / 3.0) < 1e-5


def test_unintended_penalty_binary_plus_continuous():
    from flash_chord.objectives.contact import unintended_contact_penalty

    # 2 bodies, 2 dirs. body0: demo+sim contact (intended). body1: demo none, sim has support [0.5,0.3].
    cmd = [1.0, 0.0, 0.0, 0.0]  # body0=[1,0] active, body1=[0,0] inactive
    cur = [1.0, 0.0, 0.5, 0.3]  # body1 has unintended support
    r = _launch(unintended_contact_penalty, cmd, cur, 1, 2, 2)
    binary = 1.0 / 2.0  # 1 of 2 bodies has unintended contact
    cont = (0.5**2 + 0.3**2) / 2.0 / 1.0  # mean sq support over K, over 1 inactive body
    assert abs(r - (binary + cont)) < 1e-5
