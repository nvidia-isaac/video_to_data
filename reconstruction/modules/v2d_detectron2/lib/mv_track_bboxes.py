from pathlib import Path

import torch

from v2d.mv.rig import RigConfig
from v2d.mv.io.video import FrameSource

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

    if cfg.image_dir is None and cfg.video_dir is None:
        raise ValueError("At least one of image_dir or video_dir is required")

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

        if cfg.image_dir is not None:
            frame_source = FrameSource(image_dir=Path(cfg.image_path_template.format(cam_name=cam.name)))
        elif cfg.video_dir is not None:
            frame_source = FrameSource(video_path=Path(cfg.video_path_template.format(cam_name=cam.name)))

        output_path = Path(cfg.output_path_template.format(cam_name=cam.name))

        track_bboxes(
            frame_source=frame_source,
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

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_dir", type=str, default=None, help="Directory containing images")
    input_group.add_argument("--video_dir", type=str, default=None, help="Directory containing videos")

    parser.add_argument("--weights_dir", type=str, required=True, help="Directory containing model weights")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for outputs")
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(Path(__file__).parent / "mv_track_bboxes.yaml"),
        help="Path to mv_track_bboxes.yaml (default: module's built-in config)",
    )
    parser.add_argument("--debug", type=int, default=None, help="Debug level override")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    overrides: dict = {
        "detector": {"weights_dir": args.weights_dir},
        "output_dir": args.output_dir,
    }
    if args.image_dir is not None:
        overrides["image_dir"] = args.image_dir
    elif args.video_dir is not None:
        overrides["video_dir"] = args.video_dir
    if args.debug is not None:
        overrides["debug"] = args.debug

    cfg = OmegaConf.merge(cfg, overrides)
    mv_track_bboxes_from_config(cfg)
