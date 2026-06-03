# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.wilor.docker._config import IMAGE_NAME, MODULES_DIR


def run_image_list_to_hands(
    images_dir: str,
    output_dir: str,
    weights_dir: str,
    bboxes_dir: str | None = None,
    dev: bool = False,
) -> None:
    inputs = {
        "images_dir":  images_dir,
        "weights_dir": weights_dir,
    }
    if bboxes_dir is not None:
        inputs["bboxes_dir"] = bboxes_dir
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.wilor.lib.image_list_to_hands",
        inputs=inputs,
        outputs={"output_dir": output_dir},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WiLoR over a folder of images")
    parser.add_argument("--images_dir",  required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--bboxes_dir", default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_image_list_to_hands(
        images_dir  = args.images_dir,
        output_dir  = args.output_dir,
        weights_dir = args.weights_dir,
        bboxes_dir  = args.bboxes_dir,
        dev         = args.dev,
    )
