# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from v2d.docker.container import run_in_container
from v2d.grounding_dino.docker._config import IMAGE_NAME, MODULES_DIR

_LIB_CONFIG = Path(__file__).parent.parent / "lib" / "mv_image_list_to_object_bboxes.yaml"


def run_mv_image_list_to_object_bboxes(
    rgb_dir: str,
    prompt_path: str,
    output_dir: str,
    model_dir: str,
    config_path: str = str(_LIB_CONFIG),
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.grounding_dino.lib.mv_image_list_to_object_bboxes",
        inputs={
            "rgb_dir": rgb_dir,
            "prompt_path": prompt_path,
            "model_dir": model_dir,
            "config_path": config_path,
        },
        outputs={"output_dir": output_dir},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={"PYTHONUNBUFFERED": "1"},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-view Grounding DINO object detection"
    )
    parser.add_argument("--rgb_dir", type=str, required=True,
                        help="Root directory containing per-camera image folders")
    parser.add_argument("--prompt_path", type=str, required=True,
                        help="Path to plain-text file containing the object prompt")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for per-camera bbox JSONs")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory with Grounding DINO weights")
    parser.add_argument("--config_path", type=str, default=str(_LIB_CONFIG),
                        help="Path to mv_image_list_to_object_bboxes.yaml")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_mv_image_list_to_object_bboxes(
        rgb_dir=args.rgb_dir,
        prompt_path=args.prompt_path,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        config_path=args.config_path,
        dev=args.dev,
    )
