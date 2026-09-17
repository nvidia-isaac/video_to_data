# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import os

from v2d.cari4d.docker._config import IMAGE_NAME, MODULES_DIR
from v2d.docker.container import run_in_container


def run_download_weights(output_dir: str, dev: bool = False) -> None:
    env = {"HF_TOKEN": os.environ["HF_TOKEN"]} if os.environ.get("HF_TOKEN") else None
    run_in_container(image=IMAGE_NAME, module="v2d.cari4d.lib.download_weights", inputs={}, outputs={"output_dir": output_dir}, dev=dev, modules_dir=MODULES_DIR, gpus=False, env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download all CARI4D inference weights")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_download_weights(args.output_dir, dev=args.dev)
