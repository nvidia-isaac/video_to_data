# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from v2d.common.datatypes import BoundingBox
from v2d.common.video import FrameSource
from v2d.mv.rig import RigConfig

from .datatypes import Sam2Prompt, Sam2Prompts
from .video_to_masks import video_to_masks


def bbox_track_to_prompts(bbox_path: str | Path) -> Sam2Prompts:
    """Create SAM2 prompts from a detectron2 bbox track file (.pt).

    Picks the frame with the highest detection score and creates a single
    box prompt at that frame with object_id=0.

    Args:
        bbox_path: Path to a .pt file with keys 'bbox_track' (N,4) and 'scores' (N,).

    Returns:
        Sam2Prompts with one box prompt.
    """
    data = torch.load(bbox_path, weights_only=False)
    scores = np.asarray(data["scores"])
    bbox_track = np.asarray(data["bbox_track"])

    best_idx = int(scores.argmax())
    x0, y0, x1, y1 = bbox_track[best_idx].tolist()

    prompt = Sam2Prompt(
        frame_index=best_idx,
        object_id=0,
        box=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
    )
    return Sam2Prompts(prompts=[prompt])


def bbox_json_to_prompts(bbox_path: str | Path) -> Sam2Prompts:
    """Create SAM2 prompts from a grounding-dino-format JSON file.

    The JSON maps frame stems to lists of detections, each with a
    'confidence' and 'box' ({x0, y0, x1, y1}).  Picks the single
    highest-confidence detection across all frames and returns it as a
    box prompt.  The frame stem is converted to an integer frame index.

    Args:
        bbox_path: Path to a .json file mapping frame stems to detection lists.

    Returns:
        Sam2Prompts with one box prompt.
    """
    with open(bbox_path) as f:
        results: dict[str, list[dict]] = json.load(f)

    best_conf = -1.0
    best_frame = 0
    best_box = None

    for stem, detections in results.items():
        frame_idx = int(stem)
        for det in detections:
            conf = det.get("confidence", 0.0)
            if conf > best_conf:
                best_conf = conf
                best_frame = frame_idx
                best_box = det["box"]

    if best_box is None:
        raise ValueError(f"No detections found in {bbox_path}")

    prompt = Sam2Prompt(
        frame_index=best_frame,
        object_id=0,
        box=BoundingBox(x0=best_box["x0"], y0=best_box["y0"],
                        x1=best_box["x1"], y1=best_box["y1"]),
    )
    return Sam2Prompts(prompts=[prompt])


def load_bbox_prompts(bbox_path: str | Path) -> Sam2Prompts:
    """Load bbox prompts from either a .pt track file or a .json detection file."""
    bbox_path = Path(bbox_path)
    if bbox_path.suffix == ".pt":
        return bbox_track_to_prompts(bbox_path)
    elif bbox_path.suffix == ".json":
        return bbox_json_to_prompts(bbox_path)
    else:
        raise ValueError(f"Unsupported bbox format: {bbox_path.suffix} (expected .pt or .json)")


def load_bbox_prompt_label(bbox_path: str | Path) -> str:
    """Return the label associated with the bbox selected as the SAM2 prompt."""
    bbox_path = Path(bbox_path)
    if bbox_path.suffix == ".pt":
        data = torch.load(bbox_path, weights_only=False)
        category_id = int(data.get("det_cat_id", 0))
        return "person" if category_id == 0 else f"class {category_id}"
    if bbox_path.suffix == ".json":
        with open(bbox_path) as f:
            results: dict[str, list[dict]] = json.load(f)
        detections = [det for frame_detections in results.values() for det in frame_detections]
        if not detections:
            raise ValueError(f"No detections found in {bbox_path}")
        best_detection = max(detections, key=lambda det: det.get("confidence", 0.0))
        return str(best_detection.get("label", "object"))
    raise ValueError(
        f"Unsupported bbox format: {bbox_path.suffix} (expected .pt or .json)"
    )


def save_bbox_prompt_visualization(
    source_path: str | Path,
    prompt: Sam2Prompt,
    output_path: str | Path,
    label: str,
) -> None:
    """Draw a labeled SAM2 bbox prompt on its corresponding RGB frame."""
    if prompt.box is None:
        raise ValueError("Cannot visualize a SAM2 prompt without a bounding box")

    source = FrameSource.from_path(source_path)
    try:
        if prompt.frame_index < 0 or prompt.frame_index >= source.n_frames:
            raise IndexError(
                f"Prompt frame {prompt.frame_index} is out of range for "
                f"{source_path} ({source.n_frames} frames)"
            )

        try:
            frame = source[prompt.frame_index]
        except RuntimeError as exc:
            if "Random access is not supported" not in str(exc):
                raise
            frame = next(
                frame
                for frame_idx, frame in enumerate(source.iter_frames())
                if frame_idx == prompt.frame_index
            )
    finally:
        source.close()

    image = Image.fromarray(np.asarray(frame)).convert("RGB")
    draw = ImageDraw.Draw(image)
    box = prompt.box
    line_width = max(2, round(min(image.size) / 300))
    green = (0, 255, 0)
    draw.rectangle(
        (box.x0, box.y0, box.x1, box.y1),
        outline=green,
        width=line_width,
    )

    text_bbox = draw.textbbox((0, 0), label, stroke_width=1)
    text_height = text_bbox[3] - text_bbox[1]
    label_x = max(0, round(box.x0))
    label_y = max(0, round(box.y0) - text_height - 3)
    draw.text(
        (label_x, label_y),
        label,
        fill=green,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def mv_videos_to_masks_from_config(cfg):
    """Run video_to_masks for each camera defined by the rig config."""
    rig = RigConfig(cfg.rig_config)

    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)
        print(f"\n=== Processing camera: {cam.name} ===")

        source_path = cfg.rgb_path_template.format(cam_name=cam.name)

        bbox_path = cfg.bbox_path_template.format(cam_name=cam.name)
        masks_dir = cfg.mask_path_template.format(cam_name=cam.name)

        prompts = load_bbox_prompts(bbox_path)
        prompt_label = load_bbox_prompt_label(bbox_path)
        best_frame = prompts.prompts[0].frame_index
        best_box = prompts.prompts[0].box
        print(f"  Bbox prompt: frame={best_frame}, "
              f"box=({best_box.x0:.0f}, {best_box.y0:.0f}, {best_box.x1:.0f}, {best_box.y1:.0f})")

        prompt_vis_path = Path(cfg.output_dir) / "prompts" / f"{cam.name}_bbox.png"
        save_bbox_prompt_visualization(
            source_path,
            prompts.prompts[0],
            prompt_vis_path,
            prompt_label,
        )
        print(f"  Bbox visualization: {prompt_vis_path}")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(prompts.to_dict(), f)
            prompts_path = f.name

        try:
            video_to_masks(
                source_path, prompts_path, masks_dir, cfg.weights_dir,
                mask_extension=cfg.get("mask_extension", ""),
            )
        finally:
            os.unlink(prompts_path)

        print(f"  Masks saved to {masks_dir}")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(
        description="Multi-view video/image to masks using SAM2 with detectron2 bbox prompts"
    )
    parser.add_argument("--bbox_dir", type=str, required=True,
                        help="Directory containing per-camera bbox_track .pt files")
    parser.add_argument("--rgb_dir", type=str, required=True,
                        help="Directory containing per-camera input frames")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for per-camera masks")
    parser.add_argument("--weights_dir", type=str, required=True,
                        help="Path to SAM2 weights directory")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config (merged on top of defaults)")
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "mv_videos_to_masks.yaml")
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))
    overrides: dict = {
        "bbox_dir": args.bbox_dir,
        "rgb_dir": args.rgb_dir,
        "output_dir": args.output_dir,
        "weights_dir": args.weights_dir,
    }

    cfg = OmegaConf.merge(cfg, overrides)
    mv_videos_to_masks_from_config(cfg)
