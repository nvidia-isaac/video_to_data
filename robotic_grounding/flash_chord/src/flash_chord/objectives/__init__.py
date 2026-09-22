# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tracking objectives — motion-imitation, object-tracking, contact-wrench-tracking.

Each objective term is a per-world scalar that BOTH the RL reward (weighted sum) and
the MPPI cost (horizon sum) consume — defined once here.
"""
