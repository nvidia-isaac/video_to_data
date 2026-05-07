"""Hand pipeline: detection → SAM2 tracking → HaMeR → depth alignment.

Steps:
  0.  AnyCalib undistortion           [optional, --undistort]
  1.  Extract frames
  2.  MoGe depth + intrinsics         [only when --run_hamer]
  2b. Stabilise intrinsics            [only when --run_hamer]
  3.  MediaPipe HandLandmarker        → hand_detections.json
  4.  Build SAM2 prompts               → sam2_prompts.json + hand_tracks.json
  5.  SAM2 mask propagation            → masks/{1,2,...}/*.png
  6.  Render prompts overlay PNG + mask overlay MP4
  7.  HaMeR per-track per-frame        [--run_hamer]
  8.  Render HaMeR mesh overlay        [--run_hamer]
  9.  Align HaMeR to depth             [--run_hamer]
  10. Render aligned HaMeR overlay     [--run_hamer]

Output directory layout (with --undistort + --run_hamer):
  <output_dir>/
  ├── anycalib/{intrinsics,distortion}.json
  │   ├── undistorted.mp4
  │   └── undistorted_intrinsics.json
  ├── frames/
  ├── depth/                       # MoGe depth PNGs
  ├── intrinsics/                  # MoGe per-frame intrinsics JSONs
  ├── intrinsics_stable.json       # stabilised single-JSON intrinsics
  ├── hand_detections.json
  ├── sam2_prompts.json
  ├── hand_tracks.json
  ├── prompts_overlay.png
  ├── masks_overlay.mp4
  ├── masks/{1,2}/                 # SAM2 hand masks per track
  ├── hamer/{1,2}/*.json           # HaMeR per-frame MANO + virtual cam
  ├── hamer_overlay.mp4
  ├── hamer_aligned/{1,2}/*.json   # depth-aligned, real-intrinsics
  └── hamer_aligned_overlay.mp4

Usage:
    python modules/v2d_pipelines/run_hand_masks.py \\
        --video_path data/my_video.mp4 \\
        --output_dir data/outputs/my_video \\
        --sam2_weights data/weights/sam2 \\
        --undistort \\
        --run_hamer

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

from v2d.anycalib.docker.run_video_to_calibration import run_video_to_calibration
from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.common.utils import extract_images
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.hamer.docker.run_align_hands import run_align_hands
from v2d.hamer.docker.run_masks_to_hands import run_masks_to_hands
from v2d.hamer.docker.run_render_hands_aligned_video import run_render_hands_aligned_video
from v2d.hamer.docker.run_render_hands_video import run_render_hands_video
from v2d.hand_detector.docker.run_image_to_hand_bboxes import run_image_to_hand_bboxes
from v2d.moge.docker.run_video_to_depth import run_video_to_depth as run_moge_depth
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
    undistort: bool = False,
    anycalib_weights: str = "data/weights/anycalib",
    moge_weights: str = "data/weights/moge",
    run_hamer: bool = False,
    hamer_weights: str | None = None,
    bbox_expansion: float = 1.7,
    mask_min_pixels: int = 256,
    dev: bool = False,
) -> None:
    video_path = os.path.abspath(video_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    frames_dir         = f"{output_dir}/frames"
    depth_dir          = f"{output_dir}/depth"
    intrinsics_dir     = f"{output_dir}/intrinsics"
    intrinsics_stable  = f"{output_dir}/intrinsics_stable.json"
    anycalib_dir       = f"{output_dir}/anycalib"
    anycalib_intr      = f"{anycalib_dir}/intrinsics.json"
    anycalib_dist      = f"{anycalib_dir}/distortion.json"
    undistorted_video  = f"{anycalib_dir}/undistorted.mp4"
    undistorted_intr   = f"{anycalib_dir}/undistorted_intrinsics.json"
    hand_detections    = f"{output_dir}/hand_detections.json"
    sam2_prompts       = f"{output_dir}/sam2_prompts.json"
    hand_tracks        = f"{output_dir}/hand_tracks.json"
    masks_dir          = f"{output_dir}/masks"
    prompts_overlay    = f"{output_dir}/prompts_overlay.png"
    masks_overlay      = f"{output_dir}/masks_overlay.mp4"
    ref_rgb            = f"{frames_dir}/{reference_frame:06d}.png"

    print(f"\n{'='*60}")
    print(f"  video           : {os.path.basename(video_path)}")
    print(f"  output          : {output_dir}")
    print(f"  reference_frame : {reference_frame}")
    print(f"  selfie          : {selfie}")
    print(f"  undistort       : {undistort}")
    print(f"  run_hamer       : {run_hamer}")
    print(f"{'='*60}\n")

    # 0. AnyCalib undistortion (optional) ------------------------------------
    if undistort:
        os.makedirs(anycalib_dir, exist_ok=True)
        if not _step("AnyCalib undistortion", os.path.exists(undistorted_video)):
            run_video_to_calibration(
                video_path                  = video_path,
                intrinsics_path             = anycalib_intr,
                distortion_path             = anycalib_dist,
                weights_path                = anycalib_weights,
                undistorted_video_path      = undistorted_video,
                undistorted_intrinsics_path = undistorted_intr,
                dev                         = dev,
            )
        video_path = undistorted_video
        print(f"  → using undistorted video: {video_path}")

    # 1. Extract frames -------------------------------------------------------
    if not _step("Extract frames", _has_files(frames_dir)):
        extract_images(video_path, frames_dir)

    # 2. MoGe depth + intrinsics + stabilise (only if HaMeR alignment will run)
    if run_hamer:
        moge_input_intrinsics = undistorted_intr if undistort else None
        if not _step("MoGe depth + intrinsics", _has_files(depth_dir)):
            run_moge_depth(
                video_path            = video_path,
                depth_folder          = depth_dir,
                intrinsics_folder     = intrinsics_dir,
                weights_path          = moge_weights,
                input_intrinsics_path = moge_input_intrinsics,
                dev                   = dev,
            )
        if not _step("Stabilise intrinsics", os.path.exists(intrinsics_stable)):
            stabilize_intrinsics(intrinsics_dir, intrinsics_stable)

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

    # 6. Optional HaMeR per-frame reconstruction -----------------------------
    hamer_dir             = f"{output_dir}/hamer"
    hamer_overlay         = f"{output_dir}/hamer_overlay.mp4"
    hamer_aligned_dir     = f"{output_dir}/hamer_aligned"
    hamer_aligned_overlay = f"{output_dir}/hamer_aligned_overlay.mp4"
    if run_hamer:
        if hamer_weights is None:
            raise ValueError("--hamer_weights is required when --run_hamer is set")
        mano_assets_root = os.path.join(hamer_weights, "_DATA", "data")

        # masks_to_hands has its own per-frame skip logic, so always invoke
        # it — re-runs short-circuit cheaply per JSON.
        if not _step("HaMeR per-frame regression", False):
            run_masks_to_hands(
                frames_dir       = frames_dir,
                masks_dir        = masks_dir,
                tracks_path      = hand_tracks,
                output_dir       = hamer_dir,
                weights_dir      = hamer_weights,
                bbox_expansion   = bbox_expansion,
                mask_min_pixels  = mask_min_pixels,
                dev              = dev,
            )

        if not _step("Render HaMeR mesh overlay", os.path.exists(hamer_overlay)):
            run_render_hands_video(
                frames_dir       = frames_dir,
                hamer_dir        = hamer_dir,
                mano_assets_root = mano_assets_root,
                output_path      = hamer_overlay,
                dev              = dev,
            )

        # Align HaMeR to MoGe depth + real intrinsics. Per-frame skip is
        # internal, so always invoke.
        if not _step("Align HaMeR to depth", False):
            run_align_hands(
                hamer_dir         = hamer_dir,
                depth_dir         = depth_dir,
                intrinsics_path   = intrinsics_stable,
                mano_assets_root  = mano_assets_root,
                output_dir        = hamer_aligned_dir,
                hand_masks_dir    = masks_dir,
                mask_min_pixels   = mask_min_pixels,
                dev               = dev,
            )

        if not _step("Render aligned HaMeR overlay",
                     os.path.exists(hamer_aligned_overlay)):
            run_render_hands_aligned_video(
                frames_dir       = frames_dir,
                aligned_dir      = hamer_aligned_dir,
                mano_assets_root = mano_assets_root,
                output_path      = hamer_aligned_overlay,
                dev              = dev,
            )

    print(f"\n{'='*60}")
    print(f"  Done.")
    if undistort:
        print(f"  anycalib        : {anycalib_dir}/")
    print(f"  hand_detections : {hand_detections}")
    print(f"  hand_tracks     : {hand_tracks}")
    print(f"  masks           : {masks_dir}/{{1..{len(detections)}}}/")
    print(f"  prompts_overlay : {prompts_overlay}")
    print(f"  masks_overlay   : {masks_overlay}")
    if run_hamer:
        print(f"  depth           : {depth_dir}/")
        print(f"  intrinsics      : {intrinsics_stable}")
        print(f"  hamer           : {hamer_dir}/{{1..{len(detections)}}}/*.json")
        print(f"  hamer_overlay   : {hamer_overlay}")
        print(f"  hamer_aligned   : {hamer_aligned_dir}/{{1..{len(detections)}}}/*.json")
        print(f"  hamer_aligned_overlay : {hamer_aligned_overlay}")
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
    p.add_argument("--undistort", action="store_true",
                   help="Run AnyCalib first to estimate intrinsics + distortion "
                        "and rebind video_path to the undistorted MP4.")
    p.add_argument("--anycalib_weights", default="data/weights/anycalib",
                   help="AnyCalib weights dir (used only with --undistort).")
    p.add_argument("--moge_weights", default="data/weights/moge",
                   help="MoGe weights dir (used only with --run_hamer).")
    p.add_argument("--run_hamer", action="store_true",
                   help="After SAM2, run HaMeR per (track, frame), align it "
                        "to MoGe depth, and render verification overlays.")
    p.add_argument("--hamer_weights", default="data/weights/hamer",
                   help="Dir containing _DATA/hamer_ckpts/ (and MANO_RIGHT.pkl "
                        "under _DATA/data/).")
    p.add_argument("--bbox_expansion", type=float, default=1.7,
                   help="Expand SAM2 mask bbox by this factor for HaMeR crop.")
    p.add_argument("--mask_min_pixels", type=int, default=256,
                   help="Skip HaMeR when SAM2 mask has fewer than this many pixels.")
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
        undistort                = args.undistort,
        anycalib_weights         = args.anycalib_weights,
        moge_weights             = args.moge_weights,
        run_hamer                = args.run_hamer,
        hamer_weights            = args.hamer_weights,
        bbox_expansion           = args.bbox_expansion,
        mask_min_pixels          = args.mask_min_pixels,
        dev                      = args.dev,
    )
