# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.wilor.docker._config import IMAGE_NAME, MODULES_DIR


def run_masks_to_hands(
    frames_dir: str,
    masks_dir: str,
    tracks_path: str,
    output_dir: str,
    weights_dir: str,
    bbox_expansion: float = 1.7,
    mask_min_pixels: int = 256,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.wilor.lib.masks_to_hands",
        inputs={
            "frames_dir":  frames_dir,
            "masks_dir":   masks_dir,
            "tracks_path": tracks_path,
            "weights_dir": weights_dir,
        },
        outputs={"output_dir": output_dir},
        extra_args={
            "bbox_expansion":  bbox_expansion,
            "mask_min_pixels": mask_min_pixels,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WiLoR per (track, frame) using SAM2 masks")
    parser.add_argument("--frames_dir",  required=True)
    parser.add_argument("--masks_dir",   required=True)
    parser.add_argument("--tracks_path", required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--bbox_expansion",  type=float, default=1.7)
    parser.add_argument("--mask_min_pixels", type=int,   default=256)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_masks_to_hands(
        frames_dir      = args.frames_dir,
        masks_dir       = args.masks_dir,
        tracks_path     = args.tracks_path,
        output_dir      = args.output_dir,
        weights_dir     = args.weights_dir,
        bbox_expansion  = args.bbox_expansion,
        mask_min_pixels = args.mask_min_pixels,
        dev             = args.dev,
    )
