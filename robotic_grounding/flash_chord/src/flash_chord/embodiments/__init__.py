# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embodiments: the cross-embodiment seam.

Each embodiment (dual Sharpa hands now; other hands / whole body later) implements
the :class:`~flash_chord.embodiments.base.Embodiment` interface — how to build the
robot into a Newton model, its DOF/body layout, and how a reference frame maps to
control targets. The base env, scene, runtime, objectives, and lifecycle depend on
this interface, never on a concrete robot.
"""
