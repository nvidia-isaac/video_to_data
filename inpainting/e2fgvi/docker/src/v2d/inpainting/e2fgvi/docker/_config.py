# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

IMAGE_NAME = "v2d_e2fgvi"
E2FGVI_ROOT = Path(__file__).resolve().parents[6]
LIB_SRC_DIR = str(E2FGVI_ROOT / "lib" / "src")
