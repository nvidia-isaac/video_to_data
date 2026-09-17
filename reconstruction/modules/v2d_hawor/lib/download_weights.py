# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download public HaWoR runtime weights.

MANO files are not downloaded here because they require MANO registration. The
HaWoR runner reuses sibling MANO assets from the shared weights tree when
available, for example data/weights/hand or data/weights/hamer.
"""

from __future__ import annotations

import argparse
import os
import subprocess


def _download(url: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"  present: {output_path}")
        return
    print(f"  downloading {url} -> {output_path}")
    subprocess.run(["wget", "-c", "-q", "--show-progress", url, "-O", output_path], check=True)


def _gdown(file_id: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"  present: {output_path}")
        return
    print(f"  downloading Google Drive file {file_id} -> {output_path}")
    subprocess.run(["gdown", "--continue", file_id, "-O", output_path], check=True)


def run_download(weights_dir: str) -> None:
    _download(
        "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt",
        os.path.join(weights_dir, "external", "detector.pt"),
    )
    _download(
        "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt",
        os.path.join(weights_dir, "hawor", "checkpoints", "hawor.ckpt"),
    )
    _download(
        "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/infiller.pt",
        os.path.join(weights_dir, "hawor", "checkpoints", "infiller.pt"),
    )
    _download(
        "https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml",
        os.path.join(weights_dir, "hawor", "model_config.yaml"),
    )
    _gdown("1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh", os.path.join(weights_dir, "external", "droid.pth"))
    _gdown("1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr", os.path.join(weights_dir, "external", "metric_depth_vit_large_800k.pth"))
    print(f"HaWoR weights ready in {weights_dir}")
    print("MANO assets are reused from sibling shared weights when available.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights_dir", required=True)
    args = parser.parse_args()
    run_download(args.weights_dir)


if __name__ == "__main__":
    main()
