# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

ASSETS_DIR = Path(__file__).parent.parent / "assets"
BODY_MODELS_DIR = ASSETS_DIR / "body_models"
if "HUMAN_MOTION_DATA_DIR" in os.environ:
    HUMAN_MOTION_DATA_DIR = Path(os.environ["HUMAN_MOTION_DATA_DIR"])
else:
    HUMAN_MOTION_DATA_DIR = ASSETS_DIR / "human_motion_data"
# Regenerable pipeline intermediates. For most datasets HUMAN_MOTION_DATA_DIR is an
# external mount, so intermediates land outside the repo anyway; for in-repo datasets
# like ego_recon it is a committed tree, and a regenerable intermediate does not
# belong there. Datasets opt in via DatasetConfig.loaded_in_intermediate.
if "ROBOTIC_GROUNDING_INTERMEDIATE_DIR" in os.environ:
    INTERMEDIATE_DATA_DIR = Path(os.environ["ROBOTIC_GROUNDING_INTERMEDIATE_DIR"])
else:
    INTERMEDIATE_DATA_DIR = ASSETS_DIR.parents[3] / ".cache"
SHARPA_WAVE_XMLS_DIR = ASSETS_DIR / "xmls" / "sharpawave"
G1_URDF_DIR = ASSETS_DIR / "urdfs" / "g1"
MESHES_DIR = ASSETS_DIR / "meshes"
