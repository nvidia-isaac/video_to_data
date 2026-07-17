# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package containing task implementations for various robotic environments."""

from isaaclab_tasks.utils import import_packages

# The blacklist prevents importing configs from sub-packages. v2d_whole_body imports
# onnxruntime at module load, which requires a libcudart not present in this image and
# would otherwise abort importing the unrelated tasks in this package.
_BLACKLIST_PKGS = ["utils", "v2d_whole_body"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
