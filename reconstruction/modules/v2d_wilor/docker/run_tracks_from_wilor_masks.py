# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.wilor.docker._config import IMAGE_NAME, MODULES_DIR


def run_tracks_from_wilor_masks(
    frames_dir: str,
    wilor_raw_dir: str,
    masks_dir: str,
    tracks_path: str,
    output_dir: str,
    mano_assets_root: str,
    min_iou: float = 0.1,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.wilor.lib.tracks_from_wilor_masks",
        inputs={
            "frames_dir":       frames_dir,
            "wilor_raw_dir":    wilor_raw_dir,
            "masks_dir":        masks_dir,
            "tracks_path":      tracks_path,
            "mano_assets_root": mano_assets_root,
        },
        outputs={"output_dir": output_dir},
        extra_args={"min_iou": min_iou},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Match per-frame WiLoR detections to SAM2 tracks via silhouette IoU"
    )
    parser.add_argument("--frames_dir",       required=True)
    parser.add_argument("--wilor_raw_dir",    required=True)
    parser.add_argument("--masks_dir",        required=True)
    parser.add_argument("--tracks_path",      required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--min_iou", type=float, default=0.1)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_tracks_from_wilor_masks(
        frames_dir       = args.frames_dir,
        wilor_raw_dir    = args.wilor_raw_dir,
        masks_dir        = args.masks_dir,
        tracks_path      = args.tracks_path,
        output_dir       = args.output_dir,
        mano_assets_root = args.mano_assets_root,
        min_iou          = args.min_iou,
        dev              = args.dev,
    )
