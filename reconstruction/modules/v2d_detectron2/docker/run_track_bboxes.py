# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.detectron2.docker._config import IMAGE_NAME, MODULES_DIR


def run_track_bboxes(
    rgb_path: str,
    weights_dir: str,
    output_path: str,
    model_size: str = "b",
    bbox_thr: float = 0.5,
    iou_threshold: float = 0.3,
    max_lost: int = 30,
    min_hits: int = 3,
    batch_size: int = 1,
    debug: int = 0,
    dev: bool = False,
) -> None:
    inputs = {
        "rgb_path": rgb_path,
        "weights_dir": weights_dir,
    }

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.detectron2.lib.track_bboxes",
        inputs=inputs,
        outputs={"output_path": output_path},
        extra_args={
            "model_size": model_size,
            "bbox_thr": bbox_thr,
            "iou_threshold": iou_threshold,
            "max_lost": max_lost,
            "min_hits": min_hits,
            "batch_size": batch_size,
            "debug": debug if debug > 0 else None,
        },
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
        env={
            "PYTHONUNBUFFERED": "1",
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run bbox tracking (single camera)")
    parser.add_argument("--rgb_path", type=str, required=True,
                        help="Path to input frames (image dir, .h5, or video file)")
    parser.add_argument("--weights_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help="Output .pt file path")
    parser.add_argument("--model_size", type=str, default="b", choices=["b", "l", "h"])
    parser.add_argument("--bbox_thr", type=float, default=0.5)
    parser.add_argument("--iou_threshold", type=float, default=0.3)
    parser.add_argument("--max_lost", type=int, default=30)
    parser.add_argument("--min_hits", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_track_bboxes(
        rgb_path=args.rgb_path,
        weights_dir=args.weights_dir,
        output_path=args.output_path,
        model_size=args.model_size,
        bbox_thr=args.bbox_thr,
        iou_threshold=args.iou_threshold,
        max_lost=args.max_lost,
        min_hits=args.min_hits,
        batch_size=args.batch_size,
        debug=args.debug,
        dev=args.dev,
    )
