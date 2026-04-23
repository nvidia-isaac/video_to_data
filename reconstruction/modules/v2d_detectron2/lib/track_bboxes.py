from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import imageio.v3 as iio
import torch
from tqdm import tqdm

from v2d.mv.io.video import FrameSource, get_video_writer

from .build_detector import Detector
from .bytetracker import ByteTracker
from .tracker import IoUTracker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DetectorConfig:
    model_size: str = "b"
    weights_dir: str = ""
    det_cat_id: int = 0
    bbox_thr: float = 0.5
    default_to_full_image: bool = True
    test_score_thresh: float = 0.25


@dataclass
class IoUTrackerConfig:
    iou_threshold: float = 0.3
    max_lost: int = 30
    min_hits: int = 3
    merge_max_gap: int = 10
    merge_iou_threshold: float = 0.5


@dataclass
class ByteTrackerConfig:
    track_thresh: float = 0.6
    det_thresh: float = 0.1
    match_thresh: float = 0.8
    second_match_thresh: float = 0.5
    max_lost: int = 30
    min_hits: int = 3
    merge_max_gap: int = 60
    merge_iou_threshold: float = 0.3


TrackerConfig = IoUTrackerConfig | ByteTrackerConfig


def track_bboxes(
    frame_source: FrameSource,
    output_path: str | Path,
    detector_cfg: DetectorConfig | None = None,
    tracker_cfg: TrackerConfig | None = None,
    batch_size: int = 1,
    debug: int = 0,
    detector: Detector | None = None,
) -> dict:
    """Run detection + tracking on a single camera's frames.

    Saves a .pt dict to *output_path* and returns it.

    Debug levels:
        0 — no debug output
        1 — print track info, save per-frame detection images (every 30th frame)
        2 — also render bbox_track overlay video
    """
    if detector_cfg is None:
        detector_cfg = DetectorConfig()
    if tracker_cfg is None:
        tracker_cfg = IoUTrackerConfig()

    use_byte = isinstance(tracker_cfg, ByteTrackerConfig)

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = frame_source.n_frames
    image_size = frame_source.image_size

    effective_bbox_thr = tracker_cfg.det_thresh if use_byte else detector_cfg.bbox_thr

    if detector is None:
        weights_path = f"{detector_cfg.weights_dir}/cascade_mask_rcnn_vitdet_{detector_cfg.model_size}"
        detector = Detector(
            name="vitdet",
            device=DEVICE,
            model_size=detector_cfg.model_size,
            path=weights_path,
            test_score_thresh=detector_cfg.test_score_thresh,
        )

    if debug > 0:
        debug_det_dir = output_path.parent / f"{output_path.stem}_detections"
        debug_det_dir.mkdir(parents=True, exist_ok=True)

    det_kwargs = dict(
        det_cat_id=detector_cfg.det_cat_id,
        bbox_thr=effective_bbox_thr,
        default_to_full_image=detector_cfg.default_to_full_image,
    )

    if use_byte:
        tracker: IoUTracker | ByteTracker = ByteTracker(
            track_thresh=tracker_cfg.track_thresh,
            match_thresh=tracker_cfg.match_thresh,
            second_match_thresh=tracker_cfg.second_match_thresh,
            max_lost=tracker_cfg.max_lost,
            min_hits=tracker_cfg.min_hits,
        )
        print(f"Using ByteTracker (track_thresh={tracker_cfg.track_thresh}, "
              f"det_thresh={tracker_cfg.det_thresh})")
    else:
        tracker = IoUTracker(
            iou_threshold=tracker_cfg.iou_threshold,
            max_lost=tracker_cfg.max_lost,
            min_hits=tracker_cfg.min_hits,
        )
        print(f"Using IoUTracker (iou_threshold={tracker_cfg.iou_threshold})")

    n_batches = (n_frames + batch_size - 1) // batch_size
    for batch_start, batch_frames in tqdm(frame_source.iter_batches(batch_size),
                                          total=n_batches, desc="Running detection"):
        start_time = time.time()
        if len(batch_frames) == 1:
            batch_results = [detector.run_detection(batch_frames[0], **det_kwargs)]
        else:
            batch_results = detector.run_detection_batch(batch_frames, **det_kwargs)

        for offset, (bboxes, scores) in enumerate(batch_results):
            frame_idx = batch_start + offset
            tracker.update(frame_idx, bboxes, scores)

            if debug > 0 and frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                tqdm.write(f"Frame {frame_idx} detection time: {elapsed:.3f}s")
                image = batch_frames[offset].copy()
                H_img, W_img = image.shape[:2]
                label = f"Frame {frame_idx}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                cv2.putText(image, label, (W_img - tw - 10, th + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                bboxes_int = bboxes.astype(int)
                for j, bbox in enumerate(bboxes_int):
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
                    cv2.putText(image, f"Score: {scores[j]:.2f}", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                iio.imwrite(debug_det_dir / f"bbox_{frame_idx:06d}.png", image)

    tracks = tracker.finalize()
    tracks = IoUTracker.merge_fragmented_tracks(
        tracks,
        max_gap=tracker_cfg.merge_max_gap,
        iou_threshold=tracker_cfg.merge_iou_threshold,
    )
    if debug > 0:
        print("=== Tracking Results ===")
        for track in tracks:
            print(track)

    primary_track = IoUTracker.select_primary_track(tracks, image_size)
    primary_track.interpolate(n_frames)
    bbox_track = primary_track.get_bboxes()
    scores = primary_track.get_scores()
    print(f"Primary bbox track: {primary_track}")

    result = {
        "det_cat_id": detector_cfg.det_cat_id,
        "scores": scores,
        "bbox_track": bbox_track,
    }
    torch.save(result, output_path)
    print(f"Saved bbox track ({n_frames} frames) to {output_path}")

    if debug > 1:
        debug_video_path = output_path.with_suffix(".mp4")
        writer = get_video_writer(debug_video_path, fps=30, crf=23)
        for i, image in tqdm(enumerate(frame_source.iter_frames()),
                             total=n_frames, desc="Rendering debug video"):
            H_img, W_img = image.shape[:2]
            label = f"Frame {i}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(image, label, (W_img - tw - 10, th + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            bbox = bbox_track[i].astype(int)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
            cv2.putText(image, f"Score: {scores[i]:.2f}", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            writer.write_frame(image)
        writer.close()
        print(f"Saved debug video to {debug_video_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run person detection + tracking on a single camera")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_dir", type=str, help="Directory of PNG images")
    input_group.add_argument("--video_path", type=str, help="Path to video file")

    parser.add_argument("--output_path", type=str, required=True, help="Output .pt file path")

    det = parser.add_argument_group("detector")
    det.add_argument("--model_size", type=str, default="b", choices=["b", "l", "h"])
    det.add_argument("--weights_dir", type=str, default="")
    det.add_argument("--det_cat_id", type=int, default=0, help="COCO category id (0=person)")
    det.add_argument("--bbox_thr", type=float, default=0.5)
    det.add_argument("--no_default_to_full_image", action="store_true")

    trk = parser.add_argument_group("tracker")
    trk.add_argument("--iou_threshold", type=float, default=0.3)
    trk.add_argument("--max_lost", type=int, default=30)
    trk.add_argument("--min_hits", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    source = FrameSource(
        image_dir=args.image_dir,
        video_path=args.video_path,
    )
    track_bboxes(
        frame_source=source,
        output_path=args.output_path,
        detector_cfg=DetectorConfig(
            model_size=args.model_size,
            weights_dir=args.weights_dir,
            det_cat_id=args.det_cat_id,
            bbox_thr=args.bbox_thr,
            default_to_full_image=not args.no_default_to_full_image,
        ),
        tracker_cfg=IoUTrackerConfig(
            iou_threshold=args.iou_threshold,
            max_lost=args.max_lost,
            min_hits=args.min_hits,
        ),
        batch_size=args.batch_size,
        debug=args.debug,
    )
