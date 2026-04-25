# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from robotic_grounding.retarget.bundle_paths import (
    ASSETS_DIR,
    get_human_motion_data_dir,
)

BODY_MODELS_DIR = ASSETS_DIR / "body_models"
HUMAN_MOTION_DATA_DIR = get_human_motion_data_dir()
SHARPA_WAVE_XMLS_DIR = ASSETS_DIR / "xmls" / "sharpawave"
G1_URDF_DIR = ASSETS_DIR / "urdfs" / "g1"
MESHES_DIR = ASSETS_DIR / "meshes"
