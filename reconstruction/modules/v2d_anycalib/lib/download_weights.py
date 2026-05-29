# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Pre-download AnyCalib weights into a torch hub cache directory.

AnyCalib auto-fetches its checkpoint from GitHub Releases on first instantiation
(`AnyCalib(model_id=...)`). The DINOv2 backbone is similarly pulled via
``torch.hub``. We pre-warm both caches under ``TORCH_HOME=output_dir`` so the
container can run offline afterwards.
"""
from __future__ import annotations

import argparse
import os

DEFAULT_MODEL_ID = "anycalib_gen"
AVAILABLE_MODEL_IDS = ("anycalib_pinhole", "anycalib_gen", "anycalib_dist", "anycalib_edit")


def download_anycalib(output_dir: str | None = None, model_id: str = DEFAULT_MODEL_ID) -> None:
    if model_id not in AVAILABLE_MODEL_IDS:
        raise ValueError(f"Unknown model_id '{model_id}'. Choices: {AVAILABLE_MODEL_IDS}")
    if output_dir is None:
        output_dir = os.environ.get("CHECKPOINT_DIR")
        if output_dir is None:
            raise ValueError("CHECKPOINT_DIR environment variable must be set")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.environ["TORCH_HOME"] = output_dir

    from anycalib import AnyCalib  # imported after TORCH_HOME is set
    print(f"Downloading AnyCalib '{model_id}' (and DINOv2 backbone) into {output_dir}")
    AnyCalib(model_id=model_id)  # side effect: torch.hub downloads both checkpoints
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AnyCalib checkpoint")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                        choices=AVAILABLE_MODEL_IDS)
    args = parser.parse_args()
    download_anycalib(output_dir=args.output_dir, model_id=args.model_id)
