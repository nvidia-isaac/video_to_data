# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from pathlib import Path

import torch

from v2d.mv.rig import RigConfig

from .build_detector import Detector
from .track_bboxes import (
    ByteTrackerConfig,
    DetectorConfig,
    IoUTrackerConfig,
    track_bboxes,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mv_track_bboxes_from_config(cfg):
    """Run detection + tracking across multiple cameras defined by a rig config."""
    rig = RigConfig(cfg.rig_config)

    detector_cfg = DetectorConfig(
        model_size=cfg.detector.model_size,
        weights_dir=cfg.detector.weights_dir,
        det_cat_id=cfg.detector.det_cat_id,
        bbox_thr=cfg.detector.bbox_thr,
        default_to_full_image=cfg.detector.default_to_full_image,
        test_score_thresh=cfg.detector.get("test_score_thresh", 0.25),
    )
    tracker_type = cfg.get("tracker_type", "iou")
    if tracker_type == "byte":
        bt = cfg.byte_tracker
        tracker_cfg = ByteTrackerConfig(
            track_thresh=bt.get("track_thresh", 0.6),
            det_thresh=bt.get("det_thresh", 0.1),
            match_thresh=bt.get("match_thresh", 0.8),
            second_match_thresh=bt.get("second_match_thresh", 0.5),
            max_lost=bt.get("max_lost", 30),
            min_hits=bt.get("min_hits", 3),
            merge_max_gap=bt.get("merge_max_gap", 60),
            merge_iou_threshold=bt.get("merge_iou_threshold", 0.3),
        )
    else:
        it = cfg.iou_tracker
        tracker_cfg = IoUTrackerConfig(
            iou_threshold=it.get("iou_threshold", 0.3),
            max_lost=it.get("max_lost", 30),
            min_hits=it.get("min_hits", 3),
            merge_max_gap=it.get("merge_max_gap", 10),
            merge_iou_threshold=it.get("merge_iou_threshold", 0.5),
        )

    weights_path = f"{detector_cfg.weights_dir}/cascade_mask_rcnn_vitdet_{detector_cfg.model_size}"
    detector = Detector(
        name="vitdet",
        device=DEVICE,
        model_size=detector_cfg.model_size,
        path=weights_path,
        test_score_thresh=detector_cfg.test_score_thresh,
    )

    batch_size = cfg.get("batch_size", 1)

    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)
        print(f"Running Detectron2 bbox tracking for camera: {cam.name}")

        rgb_path = Path(cfg.rgb_path_template.format(cam_name=cam.name))
        output_path = Path(cfg.output_path_template.format(cam_name=cam.name))

        track_bboxes(
            rgb_path=rgb_path,
            output_path=output_path,
            detector_cfg=detector_cfg,
            tracker_cfg=tracker_cfg,
            batch_size=batch_size,
            detector=detector,
            debug=cfg.debug,
        )


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Run multi-view detection + tracking")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Directory containing input frames")
    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing model weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for outputs")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config (merged on top of defaults)")
    parser.add_argument("--debug", type=int, default=None, help="Debug level override")
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "mv_track_bboxes.yaml")
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))
    overrides: dict = {
        "rgb_dir": args.rgb_dir,
        "detector": {"weights_dir": args.weights_dir},
        "output_dir": args.output_dir,
    }
    if args.debug is not None:
        overrides["debug"] = args.debug

    cfg = OmegaConf.merge(cfg, overrides)
    mv_track_bboxes_from_config(cfg)
