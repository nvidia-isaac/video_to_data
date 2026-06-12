# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os

# Base asset directory
ASSET_DIR = os.path.abspath(os.path.dirname(__file__))

# Asset subdirectories
MOTION_ASSET_DIR = os.path.join(ASSET_DIR, "motion_data")
OBJECTS_ASSET_DIR = os.path.join(ASSET_DIR, "objects")
POLICY_ASSET_DIR = os.path.join(ASSET_DIR, "policies")

# Scene config directory
SCENE_CONFIG_DIR = os.path.join(
    os.path.dirname(ASSET_DIR), "tasks", "scene_utils", "config"
)
