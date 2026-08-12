# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loader-side dataset registry: dataset name -> loader module.

This is the loader half of robotic_grounding's ``retarget/dataset_registry.py``
(which keeps the retarget/asset config). Each loader module exposes module-level
``parse_args()`` and ``main(args)``; ``run_loader`` dispatches by importing the
mapped module.
"""

LOADER_MODULES: dict[str, str] = {
    "taco": "v2d.task_library_loader.lib.taco_loader",
    "arctic": "v2d.task_library_loader.lib.arctic_loader",
    "oakink2": "v2d.task_library_loader.lib.oakink2_loader",
    "hot3d": "v2d.task_library_loader.lib.hot3d_loader",
    "h2o": "v2d.task_library_loader.lib.h2o_loader",
    "grab": "v2d.task_library_loader.lib.grab_loader",
    "dexycb": "v2d.task_library_loader.lib.dexycb_loader",
}
