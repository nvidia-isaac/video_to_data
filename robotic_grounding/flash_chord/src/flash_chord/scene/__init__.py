# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene construction from a parquet: spawn objects + support surfaces and assemble
the Newton model (embodiment + objects + collision), then replicate to N worlds.
Embodiment-agnostic.
"""
