# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.hamer.docker._config import IMAGE_NAME, MODULES_DIR


def run_align_hands(
    hamer_dir: str,
    depth_dir: str,
    intrinsics_path: str,
    mano_assets_root: str,
    output_dir: str,
    hand_masks_dir: str | None = None,
    object_masks_dir: str | None = None,
    mask_min_pixels: int = 256,
    dev: bool = False,
) -> None:
    inputs = {
        "hamer_dir":        hamer_dir,
        "depth_dir":        depth_dir,
        "intrinsics_path":  intrinsics_path,
        "mano_assets_root": mano_assets_root,
    }
    if hand_masks_dir is not None:
        inputs["hand_masks_dir"] = hand_masks_dir
    if object_masks_dir is not None:
        inputs["object_masks_dir"] = object_masks_dir
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hamer.lib.align_hands",
        inputs=inputs,
        outputs={"output_dir": output_dir},
        extra_args={"mask_min_pixels": mask_min_pixels},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Depth-align HaMeR per-frame outputs")
    parser.add_argument("--hamer_dir",        required=True)
    parser.add_argument("--depth_dir",        required=True)
    parser.add_argument("--intrinsics_path",  required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--hand_masks_dir",   default=None)
    parser.add_argument("--object_masks_dir", default=None)
    parser.add_argument("--mask_min_pixels",  type=int, default=256)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_align_hands(
        hamer_dir        = args.hamer_dir,
        depth_dir        = args.depth_dir,
        intrinsics_path  = args.intrinsics_path,
        mano_assets_root = args.mano_assets_root,
        output_dir       = args.output_dir,
        hand_masks_dir   = args.hand_masks_dir,
        object_masks_dir = args.object_masks_dir,
        mask_min_pixels  = args.mask_min_pixels,
        dev              = args.dev,
    )
