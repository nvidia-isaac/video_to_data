# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset-recording variant of the Sharpa V2D task.

Adds a fixed third-person camera to the scene and an Isaac Lab ``RecorderManager``
that writes per-episode rollouts (camera RGB/depth/segmentation, proprio states,
actions and rewards) to an HDF5 dataset. Used by
``scripts/rsl_rl/record_dataset.py``.
"""
