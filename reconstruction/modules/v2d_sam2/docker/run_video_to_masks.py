# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.sam2.docker._config import IMAGE_NAME, MODULES_DIR

def run_video_to_masks(video_path: str, prompts_path: str, masks_dir: str, weights_dir: str, dev: bool = False) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.sam2.lib.video_to_masks",
        inputs={"video_path": video_path, "prompts_path": prompts_path, "weights_dir": weights_dir},
        outputs={"masks_dir": masks_dir},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video to masks using SAM2")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--prompts_path", type=str, required=True, help="Path to prompts JSON file")
    parser.add_argument("--masks_dir", type=str, required=True, help="Output directory for masks")
    parser.add_argument("--weights_dir", type=str, required=True, help="Path to SAM2 weights directory")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_video_to_masks(args.video_path, args.prompts_path, args.masks_dir, args.weights_dir, dev=args.dev)
