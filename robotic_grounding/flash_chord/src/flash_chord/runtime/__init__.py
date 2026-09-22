# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rollout runtime — shared by MPPI and RL.

Action computation (reference + residual -> embodiment targets), the Virtual Object
Controller (PD wrench on the object root; scaled implicit drives for articulations), and
contact sensing (Newton contacts -> per (hand-link, object-body) points/normals/forces).
"""
