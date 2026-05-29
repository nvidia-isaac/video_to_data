# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from v2d.docker.container import run_in_container
from v2d.nlf.docker._config import IMAGE_NAME, MODULES_DIR


def run_render_smpl_overlay(
    video_path: str,
    smpl_params_path: str,
    intrinsics_path: str,
    output_dir: str,
    weights_dir: str,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.nlf.lib.render_smpl_overlay",
        inputs={"video_path": video_path, "smpl_params_path": smpl_params_path, "intrinsics_path": intrinsics_path, "weights_dir": weights_dir},
        outputs={"output_dir": output_dir},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SMPL overlay rendering in Docker")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--smpl_params_path", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_render_smpl_overlay(
        args.video_path, args.smpl_params_path, args.intrinsics_path,
        args.output_dir, args.weights_dir, dev=args.dev,
    )
