# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
HOI object reconstruction: generate mask for the reference frame only.

Uses SAM2 image predictor (single-frame, no video propagation) to produce
the mask needed by FoundationPose's register() call.

Usage:
    python image_to_mask.py \
        --image_path /data/book_1/left/000000.jpg \
        --prompts_path /data/book_1/prompts.json \
        --masks_dir /data/book_1/masks
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from v2d.sam2.lib.datatypes import Sam2Prompts


def _build_predictor(checkpoint_dir):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import hydra

    config_file = "sam2.1_hiera_l.yaml"
    checkpoint_file = "sam2.1_hiera_large.pt"

    os.environ["HYDRA_CONFIG_SEARCH_PATH"] = checkpoint_dir
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_file)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=checkpoint_dir, job_name="sam2"):
        cfg = compose(config_name=config_file)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)

    with open(ckpt_path, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")
    model.load_state_dict(state_dict["model"], strict=True)
    model = model.to("cuda").eval()

    return SAM2ImagePredictor(model)


def mask_reference_frame(image_path: str, prompts_path: str, masks_dir: str):
    data_dir = os.environ.get("DATA_DIR", "/data")
    checkpoint_dir = os.environ.get(
        "CHECKPOINT_DIR",
        os.path.join(data_dir, "sam2/checkpoints/sam2.1-hiera-large")
    )

    with open(prompts_path) as f:
        prompts = Sam2Prompts.from_dict(json.load(f))

    # Find prompt for reference frame (frame_index 0 or the first prompt)
    prompt = prompts.prompts[0]
    points = np.array([[p.x, p.y] for p in prompt.points]) if prompt.points else None
    point_labels = np.array(prompt.point_labels) if prompt.point_labels else None
    box = np.array([prompt.box.x0, prompt.box.y0, prompt.box.x1, prompt.box.y1]) \
        if prompt.box else None

    image = np.array(Image.open(image_path).convert("RGB"))

    predictor = _build_predictor(checkpoint_dir)
    with torch.inference_mode():
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            box=box,
            multimask_output=False,
        )

    # masks shape: (1, H, W) — take best mask
    mask = masks[0].astype(bool)

    obj_id = prompt.object_id
    obj_dir = os.path.join(masks_dir, str(obj_id))
    os.makedirs(obj_dir, exist_ok=True)

    frame_idx = prompt.frame_index
    out_path = os.path.join(obj_dir, f"{frame_idx:06d}.png")
    Image.fromarray((mask * 255).astype("uint8"), mode="L").save(out_path)
    print(f"Saved mask to {out_path} (score: {scores[0]:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reference frame mask using SAM2 image predictor")
    parser.add_argument("--image_path", required=True, help="Path to reference frame image")
    parser.add_argument("--prompts_path", required=True, help="Path to prompts.json")
    parser.add_argument("--masks_dir", required=True, help="Output masks directory")
    args = parser.parse_args()
    mask_reference_frame(args.image_path, args.prompts_path, args.masks_dir)
