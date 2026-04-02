# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent / "assets"
BODY_MODELS_DIR = ASSETS_DIR / "body_models"
if "HUMAN_MOTION_DATA_DIR" in os.environ:
    HUMAN_MOTION_DATA_DIR = Path(os.environ["HUMAN_MOTION_DATA_DIR"])
else:
    HUMAN_MOTION_DATA_DIR = ASSETS_DIR / "human_motion_data"
SHARPA_WAVE_XMLS_DIR = ASSETS_DIR / "xmls" / "sharpawave"
G1_URDF_DIR = ASSETS_DIR / "urdfs" / "g1"
MESHES_DIR = ASSETS_DIR / "meshes"
