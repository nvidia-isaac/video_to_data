# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from v2d.docker.container import run_in_container
from v2d.nlf.docker._config import IMAGE_NAME, MODULES_DIR


def run_video_to_smpl(
    video_path: str,
    masks_dir: str,
    intrinsics_path: str,
    gender: str,
    output_path: str,
    weights_dir: str,
    model_type: str = "smplh",
    chunk_size: int = 32,
    dev: bool = False,
) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.nlf.lib.video_to_smpl",
        inputs={"video_path": video_path, "masks_dir": masks_dir, "intrinsics_path": intrinsics_path, "weights_dir": weights_dir},
        outputs={"output_path": output_path},
        extra_args={"gender": gender, "model_type": model_type, "chunk_size": chunk_size},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NLF video to SMPL in Docker")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--gender", required=True, choices=["male", "female", "neutral"])
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--weights_dir", required=True)
    parser.add_argument("--model_type", default="smplh", choices=["smpl", "smplh"])
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_video_to_smpl(
        args.video_path, args.masks_dir, args.intrinsics_path, args.gender,
        args.output_path, args.weights_dir, model_type=args.model_type,
        chunk_size=args.chunk_size, dev=args.dev,
    )
