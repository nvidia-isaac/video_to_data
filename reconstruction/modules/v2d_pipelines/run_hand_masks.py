"""Hand-mask pipeline: detect hands on a reference frame and propagate via SAM2.

Steps:
  1. Extract frames from the input video.
  2. Run MediaPipe Hands on the reference frame  → hand_detections.json
  3. Build SAM2 prompts from those bboxes        → sam2_prompts.json
  4. Run SAM2 video propagation                  → masks/{1,2,...}/*.png
  5. Write a sidecar mapping object_id → handedness → hand_tracks.json
  6. Render prompts overlay PNG + mask overlay MP4 for verification.

Output directory layout:
  <output_dir>/
  ├── frames/                   # extracted video frames
  ├── hand_detections.json      # MediaPipe per-detection list
  ├── sam2_prompts.json         # one prompt per hand, object_id = 1..N
  ├── hand_tracks.json          # {object_id: is_right, score} mapping
  ├── prompts_overlay.png       # reference frame + bbox prompts
  ├── masks_overlay.mp4         # full video + per-track colored masks
  └── masks/
      ├── 1/000000.png ...
      └── 2/000000.png ...

Usage:
    python modules/v2d_pipelines/run_hand_masks.py \\
        --video_path data/my_video.mp4 \\
        --output_dir data/outputs/my_video \\
        --sam2_weights data/weights/sam2

Run from reconstruction/.
"""

import argparse
import glob
import json
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.common.utils import extract_images
from v2d.hand_detector.docker.run_image_to_hand_bboxes import run_image_to_hand_bboxes
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks


# Distinct colors for up to ~10 tracks. Track id 1 → palette[0], etc.
_TRACK_COLORS = [
    (255,  60,  60),   # red
    ( 60, 160, 255),   # blue
    ( 60, 200,  60),   # green
    (255, 180,  40),   # orange
    (200,  80, 220),   # purple
    ( 60, 220, 220),   # cyan
    (255, 240,  60),   # yellow
    (240, 100, 180),   # pink
    (180, 220, 100),   # lime
    (140, 110,  80),   # brown
]


def _track_color(object_id: int) -> tuple[int, int, int]:
    return _TRACK_COLORS[(object_id - 1) % len(_TRACK_COLORS)]


def _font(size: int = 18):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _render_prompts_overlay(
    image_path: str, tracks: list[dict], detections: list[dict], output_path: str,
) -> None:
    """Save reference frame with bbox prompts drawn per track."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(20)
    for track, det in zip(tracks, detections):
        oid   = track["object_id"]
        color = _track_color(oid)
        bb    = det["bbox"]
        x0, y0, x1, y1 = bb["x0"], bb["y0"], bb["x1"], bb["y1"]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        label = f"id={oid}  {'R' if track['is_right'] else 'L'}  {track['score']:.2f}"
        tw = draw.textlength(label, font=font)
        draw.rectangle([x0, y0 - 26, x0 + tw + 8, y0], fill=color)
        draw.text((x0 + 4, y0 - 24), label, fill=(0, 0, 0), font=font)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    img.save(output_path)
    print(f"  Wrote prompts overlay   → {output_path}")


def _render_masks_overlay(
    frames_dir: str,
    masks_dir: str,
    tracks: list[dict],
    output_path: str,
    fps: float = 30.0,
    alpha: float = 0.45,
) -> None:
    """Encode a video with per-track colored mask overlays on every frame."""
    frame_files = sorted(
        glob.glob(os.path.join(frames_dir, "*.png")) +
        glob.glob(os.path.join(frames_dir, "*.jpg"))
    )
    if not frame_files:
        raise FileNotFoundError(f"No frames in {frames_dir}")

    track_ids = [t["object_id"] for t in tracks]
    label_font = _font(18)

    with tempfile.TemporaryDirectory() as tmpdir:
        for f_idx, frame_path in enumerate(frame_files):
            frame = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32)
            for oid in track_ids:
                mask_path = os.path.join(masks_dir, str(oid), f"{f_idx:06d}.png")
                if not os.path.exists(mask_path):
                    continue
                mask = np.asarray(Image.open(mask_path), dtype=np.float32)
                if mask.ndim == 3:
                    mask = mask[..., 0]
                if mask.max() <= 1.0:
                    mask = mask * 255.0
                m = (mask > 127).astype(np.float32)[..., None]   # (H, W, 1)
                color = np.array(_track_color(oid), dtype=np.float32)[None, None, :]
                frame = frame * (1 - alpha * m) + color * (alpha * m)
            out_img = Image.fromarray(frame.clip(0, 255).astype(np.uint8))
            # Legend: small color swatches + "id=N (L|R)" labels in the corner
            draw = ImageDraw.Draw(out_img)
            for i, t in enumerate(tracks):
                y = 8 + i * 24
                color = _track_color(t["object_id"])
                draw.rectangle([8, y, 28, y + 20], fill=color)
                draw.text((34, y), f"id={t['object_id']}  {'R' if t['is_right'] else 'L'}",
                          fill=(255, 255, 255), font=label_font,
                          stroke_width=2, stroke_fill=(0, 0, 0))
            out_img.save(os.path.join(tmpdir, f"{f_idx:06d}.png"))

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-r", str(fps),
            "-i", os.path.join(tmpdir, "%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
            output_path,
        ], check=True)
    print(f"  Wrote masks overlay     → {output_path}")


def _has_files(directory: str) -> bool:
    return os.path.isdir(directory) and bool(os.listdir(directory))


def _step(label: str, done: bool) -> bool:
    if done:
        print(f"  [skip] {label}")
        return True
    print(f"  [run ] {label}")
    return False


def run_hand_masks(
    video_path: str,
    output_dir: str,
    sam2_weights: str,
    reference_frame: int = 0,
    selfie: bool = False,
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    pad_ratio: float = 0.15,
    dev: bool = False,
) -> None:
    video_path = os.path.abspath(video_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    frames_dir       = f"{output_dir}/frames"
    hand_detections  = f"{output_dir}/hand_detections.json"
    sam2_prompts     = f"{output_dir}/sam2_prompts.json"
    hand_tracks      = f"{output_dir}/hand_tracks.json"
    masks_dir        = f"{output_dir}/masks"
    prompts_overlay  = f"{output_dir}/prompts_overlay.png"
    masks_overlay    = f"{output_dir}/masks_overlay.mp4"
    ref_rgb          = f"{frames_dir}/{reference_frame:06d}.png"

    print(f"\n{'='*60}")
    print(f"  video           : {os.path.basename(video_path)}")
    print(f"  output          : {output_dir}")
    print(f"  reference_frame : {reference_frame}")
    print(f"  selfie          : {selfie}")
    print(f"{'='*60}\n")

    # 1. Extract frames -------------------------------------------------------
    if not _step("Extract frames", _has_files(frames_dir)):
        extract_images(video_path, frames_dir)

    # 2. Hand detection on reference frame ------------------------------------
    # Treat an existing-but-empty detections file as "redo" — otherwise a
    # cached failure (no hands found) would short-circuit every later run.
    cached_nonempty = (
        os.path.exists(hand_detections)
        and os.path.getsize(hand_detections) > len(b"[]\n")
    )
    if not _step("Hand detection (MediaPipe)", cached_nonempty):
        run_image_to_hand_bboxes(
            image_path               = ref_rgb,
            output_path              = hand_detections,
            max_num_hands            = max_num_hands,
            min_detection_confidence = min_detection_confidence,
            pad_ratio                = pad_ratio,
            selfie                   = selfie,
            dev                      = dev,
        )

    with open(hand_detections) as f:
        detections = json.load(f)
    if not detections:
        raise RuntimeError(
            f"Hand detector found no hands in {ref_rgb}.  Things to try:\n"
            f"  • lower --min_detection_confidence (current: {min_detection_confidence})\n"
            f"  • pick a different --reference_frame where hands are more visible\n"
            f"  • toggle --selfie if footage is mirrored\n"
            f"MediaPipe Hands is weak on egocentric views; a different detector "
            f"may be needed.  Delete {hand_detections} before retrying."
        )

    # 3. Build SAM2 prompts ---------------------------------------------------
    # object_id allocation: 1..N in detection-confidence order. Handedness is
    # not part of SAM2's schema — it lives in hand_tracks.json alongside.
    if not (os.path.exists(sam2_prompts) and os.path.exists(hand_tracks)):
        prompts: list[Sam2Prompt] = []
        track_meta: list[dict] = []
        for i, det in enumerate(detections):
            object_id = i + 1
            box = BoundingBox.from_dict(det["bbox"])
            prompts.append(Sam2Prompt(
                frame_index=reference_frame,
                object_id=object_id,
                box=box,
            ))
            track_meta.append({
                "object_id": object_id,
                "role":      "hand",
                "is_right":  bool(det["is_right"]),
                "score":     float(det["score"]),
            })
        Sam2Prompts(prompts=prompts).save(sam2_prompts)
        with open(hand_tracks, "w") as f:
            json.dump({
                "reference_frame": reference_frame,
                "tracks":          track_meta,
            }, f, indent=2)
        print(f"  Wrote {len(prompts)} prompt(s) → {sam2_prompts}")
        print(f"  Wrote track metadata     → {hand_tracks}")

    # 4. SAM2 video propagation -----------------------------------------------
    # Already-propagated runs are detected by the presence of any mask subdir.
    sam2_done = any(
        _has_files(os.path.join(masks_dir, d))
        for d in (os.listdir(masks_dir) if os.path.isdir(masks_dir) else [])
    )
    if not _step("SAM2 mask propagation", sam2_done):
        run_video_to_masks(
            video_path  = video_path,
            prompts_path= sam2_prompts,
            masks_dir   = masks_dir,
            weights_dir = sam2_weights,
            dev         = dev,
        )

    # 5. Verification renders -------------------------------------------------
    with open(hand_tracks) as f:
        track_meta = json.load(f)["tracks"]

    if not _step("Render prompts overlay", os.path.exists(prompts_overlay)):
        _render_prompts_overlay(ref_rgb, track_meta, detections, prompts_overlay)

    if not _step("Render masks overlay video", os.path.exists(masks_overlay)):
        _render_masks_overlay(
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            tracks=track_meta,
            output_path=masks_overlay,
        )

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"  hand_detections : {hand_detections}")
    print(f"  hand_tracks     : {hand_tracks}")
    print(f"  masks           : {masks_dir}/{{1..{len(detections)}}}/")
    print(f"  prompts_overlay : {prompts_overlay}")
    print(f"  masks_overlay   : {masks_overlay}")
    print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_path",   required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--sam2_weights", default="data/weights/sam2")
    p.add_argument("--reference_frame", type=int, default=0)
    p.add_argument("--selfie", action="store_true",
                   help="Input is from a selfie/front-facing camera (mirrored). "
                        "Defaults to rear/ego convention.")
    p.add_argument("--max_num_hands", type=int, default=2)
    p.add_argument("--min_detection_confidence", type=float, default=0.3,
                   help="MediaPipe detection threshold (default 0.3 — lowered "
                        "from MediaPipe's stock 0.5 for harder views).")
    p.add_argument("--pad_ratio", type=float, default=0.15)
    p.add_argument("--dev", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_hand_masks(
        video_path               = args.video_path,
        output_dir               = args.output_dir,
        sam2_weights             = args.sam2_weights,
        reference_frame          = args.reference_frame,
        selfie                   = args.selfie,
        max_num_hands            = args.max_num_hands,
        min_detection_confidence = args.min_detection_confidence,
        pad_ratio                = args.pad_ratio,
        dev                      = args.dev,
    )
