# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from v2d.docker.container import run_in_container
from v2d.nlf.docker._config import IMAGE_NAME, MODULES_DIR


def run_align_depth_to_smpl(
    depth_folder: str,
    smpl_depth_folder: str,
    output_depth_folder: str,
    masks_folder: str,
    smpl_masks_folder: str = None,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.nlf.lib.align_depth_to_smpl",
        inputs={"depth_folder": depth_folder, "smpl_depth_folder": smpl_depth_folder, "masks_folder": masks_folder, "smpl_masks_folder": smpl_masks_folder},
        outputs={"output_depth_folder": output_depth_folder},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run depth-to-SMPL alignment in Docker")
    parser.add_argument("--depth_folder", required=True)
    parser.add_argument("--smpl_depth_folder", required=True)
    parser.add_argument("--output_depth_folder", required=True)
    parser.add_argument("--masks_folder", required=True)
    parser.add_argument("--smpl_masks_folder", default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_align_depth_to_smpl(
        args.depth_folder, args.smpl_depth_folder, args.output_depth_folder,
        args.masks_folder, smpl_masks_folder=args.smpl_masks_folder, dev=args.dev,
    )
