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

# Whole-body example sequence shipped with the repo. Used as the reference
# motion by the whole-body environment tests.
WHOLE_BODY_EXAMPLE_MOTION = os.path.join(
    ASSET_DIR,
    "human_motion_data",
    "whole_body",
    "soma",
    "sequence_id=2026-03-06_10-24-18_snack_box_pick_and_place_01",
    "robot_name=g1",
)
