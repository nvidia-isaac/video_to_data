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

from v2d.common.datatypes import BoundingBox
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
        best_frame = prompts.prompts[0].frame_index
        best_box = prompts.prompts[0].box
        print(f"  Bbox prompt: frame={best_frame}, "
              f"box=({best_box.x0:.0f}, {best_box.y0:.0f}, {best_box.x1:.0f}, {best_box.y1:.0f})")

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
