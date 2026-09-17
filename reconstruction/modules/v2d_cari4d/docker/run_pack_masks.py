# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

from v2d.cari4d.docker._config import IMAGE_NAME, MODULES_DIR
from v2d.docker.container import run_in_container


def run_pack_masks(video_path: str, human_masks_path: str, object_masks_path: str, output_path: str, *, overwrite: bool = False, dev: bool = False) -> None:
    run_in_container(image=IMAGE_NAME, module="v2d.cari4d.lib.pack_masks", inputs={"video_path": video_path, "human_masks_path": human_masks_path, "object_masks_path": object_masks_path}, outputs={"output_path": output_path}, extra_args={"overwrite": overwrite}, dev=dev, modules_dir=MODULES_DIR, gpus=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack SAM2 human and object masks into CARI4D H5 inside Docker")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--human_masks_path", required=True)
    parser.add_argument("--object_masks_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dev", action="store_true")
    run_pack_masks(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
