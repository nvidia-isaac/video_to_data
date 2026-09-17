# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop GR00T inference: transport, embodiment profiles, and the action adapter.

Generic spine (write once): ``policy_client`` (ZMQ transport), ``embodiment`` +
``action_adapter`` (obs translation + server-driven action reassembly + horizon buffer).
Per-embodiment plugs live under ``profiles/`` (+ a matching IsaacLab absolute action term
and inference env cfg). See ``README.md``.
"""
