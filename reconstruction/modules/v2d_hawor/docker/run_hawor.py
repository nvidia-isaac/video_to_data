# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.hawor.docker._config import IMAGE_NAME, MODULES_DIR


def run_hawor(
    video: str,
    hand_tracks_dir: str,
    weights: str,
    native_dir: str | None = None,
    focal_length: float = -1.0,
    left_id: int = 2,
    right_id: int = 3,
    max_num: int = 1000,
    dev: bool = False,
) -> None:
    outputs = {"hand_tracks_dir": hand_tracks_dir}
    if native_dir is not None:
        outputs["native_dir"] = native_dir
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.hawor.lib.run_hawor",
        inputs={"video": video, "weights": weights},
        outputs=outputs,
        extra_args={
            "focal_length": focal_length,
            "left_id": left_id,
            "right_id": right_id,
            "max_num": max_num,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HaWoR and export canonical v2d hand tracks")
    parser.add_argument("--video", required=True)
    parser.add_argument("--hand_tracks_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--native_dir", default=None)
    parser.add_argument("--focal_length", type=float, default=-1.0)
    parser.add_argument("--left_id", type=int, default=2)
    parser.add_argument("--right_id", type=int, default=3)
    parser.add_argument("--max_num", type=int, default=1000)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_hawor(
        video=args.video,
        hand_tracks_dir=args.hand_tracks_dir,
        weights=args.weights,
        native_dir=args.native_dir,
        focal_length=args.focal_length,
        left_id=args.left_id,
        right_id=args.right_id,
        max_num=args.max_num,
        dev=args.dev,
    )
