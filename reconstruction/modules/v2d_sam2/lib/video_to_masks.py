"""
SAM2 video to masks processing function.
Can be called directly from command line or imported as a function.
"""
from v2d.sam2.lib.sam2_utils import build_sam2_video_predictor_low_mem
from v2d.sam2.lib.datatypes import Sam2Prompts
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from v2d.common.video import FrameWriter

_predictor = None

def _get_predictor(weights_dir: str):
    global _predictor
    if _predictor is None:
        config_file = os.environ.get("CONFIG_FILE", "configs/sam2.1/sam2.1_hiera_l.yaml")
        checkpoint_file = os.environ.get("CHECKPOINT_FILE", "sam2.1_hiera_large.pt")

        os.environ["HYDRA_CONFIG_SEARCH_PATH"] = weights_dir
        ckpt_path = os.path.join(weights_dir, checkpoint_file)
        _predictor = build_sam2_video_predictor_low_mem(
            config_file=config_file,
            ckpt_path=ckpt_path,
        )
    return _predictor

def video_to_masks(
    video_path: str,
    prompts_path: str,
    masks_dir: str,
    weights_dir: str,
    mask_extension: str = "",
):
    """Process a video with SAM2 prompts and save masks to files.

    Per object id, the writer location is `<masks_dir>/<obj_id><mask_extension>`.
    Default `mask_extension=""` writes a PNG directory; `".h5"` writes a single
    HDF5 file (auto-detected by `FrameWriter.from_path` via the suffix).

    Runs under ``torch.autocast(bfloat16)`` because SAM2 stores its memory bank
    in bfloat16 (``sam2_utils.py:1026,1078``) — without autocast, multi-frame
    prompts (prompts at frame_idx > 0) hit a dtype mismatch when the cross-
    attention reads the memory through fp32 linear layers. Pass 1 happens to
    skirt the issue with a single ref-frame prompt; pass 2 triggers it.
    """
    with open(prompts_path, "r") as f:
        prompts = Sam2Prompts.from_dict(json.load(f))

    predictor = _get_predictor(weights_dir)

    obj_frames: dict[int, dict[int, np.ndarray]] = {}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        inference_state = predictor.init_state(video_path)

        for prompt in prompts.prompts:
            if prompt.mask_path is not None:
                # Per-frame mask prompt — uint8 PNG (>0 = foreground). SAM2's
                # add_new_mask resizes internally to the model's image size.
                mask_np = np.asarray(Image.open(prompt.mask_path)) > 0
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=prompt.frame_index,
                    obj_id=prompt.object_id,
                    mask=mask_np,
                )
                continue
            box = [prompt.box.x0, prompt.box.y0, prompt.box.x1, prompt.box.y1] if prompt.box else None
            points = [[p.x, p.y] for p in prompt.points] if prompt.points else None
            point_labels = prompt.point_labels if prompt.point_labels else None
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=prompt.frame_index,
                obj_id=prompt.object_id,
                points=points,
                labels=point_labels,
                box=box,
            )

        for reverse in [True, False]:
            for frame_idx, object_ids, masks in predictor.propagate_in_video(inference_state, reverse=reverse):
                for i, obj_id in enumerate(object_ids):
                    mask_data = (masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                    obj_frames.setdefault(obj_id, {})[frame_idx] = mask_data

    masks_path = Path(masks_dir)

    for obj_id, frames_dict in obj_frames.items():
        out = masks_path / f"{obj_id}{mask_extension}"
        writer = FrameWriter.from_path(out)
        for fidx in sorted(frames_dict.keys()):
            writer.write_frame(frames_dict[fidx], stem=f"{fidx:06d}")
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video to masks using SAM2")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--prompts_path", type=str, required=True, help="Path to prompts JSON file")
    parser.add_argument("--masks_dir", type=str, required=True, help="Output directory for masks")
    parser.add_argument("--weights_dir", type=str, required=True, help="Path to SAM2 weights directory")

    args = parser.parse_args()
    video_to_masks(args.video_path, args.prompts_path, args.masks_dir, args.weights_dir)
