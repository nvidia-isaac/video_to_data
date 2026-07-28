# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.wilor.docker._config import IMAGE_NAME, MODULES_DIR


def run_render_wilor_prompt_masks(
    wilor_json: str,
    image_path: str,
    output_dir: str,
    mano_assets_root: str,
    frame_index: int = 0,
    first_object_id: int = 1,
    min_mask_pixels: int = 1,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.wilor.lib.render_prompt_masks",
        inputs={
            "wilor_json":       wilor_json,
            "image_path":       image_path,
            "mano_assets_root": mano_assets_root,
        },
        outputs={"output_dir": output_dir},
        extra_args={
            "frame_index":     frame_index,
            "first_object_id": first_object_id,
            "min_mask_pixels": min_mask_pixels,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render WiLoR MANO silhouettes as SAM2 mask prompts"
    )
    parser.add_argument("--wilor_json", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mano_assets_root", required=True)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--first_object_id", type=int, default=1)
    parser.add_argument("--min_mask_pixels", type=int, default=1)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_render_wilor_prompt_masks(
        wilor_json=args.wilor_json,
        image_path=args.image_path,
        output_dir=args.output_dir,
        mano_assets_root=args.mano_assets_root,
        frame_index=args.frame_index,
        first_object_id=args.first_object_id,
        min_mask_pixels=args.min_mask_pixels,
        dev=args.dev,
    )
