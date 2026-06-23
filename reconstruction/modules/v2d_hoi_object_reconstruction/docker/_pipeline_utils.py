# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for HOI pipeline orchestration scripts."""

import subprocess
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# GPU memory thresholds (MiB)
_DEPTH_GPU_MEM_MIB      = 16 * 1024
_GPU_BUSY_THRESHOLD_MIB =  4 * 1024
_GPU_BUSY_BUFFER_MIB    = 35 * 1024
_GPU_IDLE_BUFFER_MIB    =  2 * 1024


def detect_gpu_ids(mem_required_mib: int = _DEPTH_GPU_MEM_MIB) -> list:
    """Return list of GPU IDs sized by available memory capacity."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [0]
    gpu_ids = []
    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) == 3:
            idx  = int(parts[0].strip())
            free = int(parts[1].strip())
            used = int(parts[2].strip())
            buf    = _GPU_BUSY_BUFFER_MIB if used > _GPU_BUSY_THRESHOLD_MIB else _GPU_IDLE_BUFFER_MIB
            usable = max(0, free - buf)
            slots  = usable // mem_required_mib
            gpu_ids.extend([idx] * slots)
    return gpu_ids if gpu_ids else [0]


def count_images(directory) -> int:
    """Count image files in a directory."""
    return sum(1 for p in Path(directory).iterdir()
               if p.suffix.lower() in IMAGE_EXTENSIONS)
