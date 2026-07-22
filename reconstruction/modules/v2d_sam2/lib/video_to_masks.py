# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SAM2 video to masks processing function.
Can be called directly from command line or imported as a function.
"""

from v2d.sam2.lib.sam2_utils import build_sam2_video_predictor_low_mem
from v2d.sam2.lib.datatypes import Sam2Prompts
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from v2d.common.video import FrameWriter
from v2d.sam2.lib.generation import (
    build_static_identity,
    commit_generation,
    validate_generation,
)

_predictor = None


def _resolve_prompt_mask_path(mask_path: str, prompts_path: str) -> Path:
    path = Path(mask_path)
    if not path.is_absolute():
        path = Path(prompts_path).resolve().parent / path
    return path


def _validate_prompt(prompt) -> None:
    prompt_types = sum(
        [
            bool(prompt.mask_path),
            prompt.box is not None,
            bool(prompt.points),
        ]
    )
    if prompt_types != 1:
        raise ValueError(
            "Each SAM2 prompt must provide exactly one of mask_path, box, or "
            f"points. Got object_id={prompt.object_id}, "
            f"frame_index={prompt.frame_index}."
        )


def _get_predictor(weights_dir: str):
    global _predictor
    if _predictor is None:
        config_file = os.environ.get(
            "CONFIG_FILE", "configs/sam2.1/sam2.1_hiera_l.yaml"
        )
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
    *,
    image_id: str | None = None,
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
        prompt_data = json.load(f)
    prompts = Sam2Prompts.from_dict(prompt_data)
    for prompt in prompts.prompts:
        _validate_prompt(prompt)

    strict_generation = image_id is not None
    if strict_generation and mask_extension:
        raise ValueError("Strict single-video SAM2 generations require PNG output")
    output = Path(masks_dir).resolve()
    if strict_generation:
        static_identity = build_static_identity(
            video_path,
            prompts_path,
            weights_dir,
            prompt_data.get("prompts") or [],
            image_id,
        )
        if output.exists():
            if not output.is_dir():
                raise FileExistsError(f"SAM2 output path is not a directory: {output}")
            return validate_generation(output, static_identity)
        if os.path.lexists(output):
            raise FileExistsError(f"SAM2 output path is not a directory: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output.name}.", suffix=".partial", dir=output.parent
            )
        )
    else:
        static_identity = None
        write_root = output

    try:
        predictor = _get_predictor(weights_dir)

        obj_frames: dict[int, dict[int, np.ndarray]] = {}
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            inference_state = predictor.init_state(video_path)

            for prompt in prompts.prompts:
                if prompt.mask_path:
                    mask_path = _resolve_prompt_mask_path(
                        prompt.mask_path, prompts_path
                    )
                    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
                    predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=prompt.frame_index,
                        obj_id=prompt.object_id,
                        mask=mask,
                    )
                    continue

                box = (
                    [prompt.box.x0, prompt.box.y0, prompt.box.x1, prompt.box.y1]
                    if prompt.box
                    else None
                )
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
                for frame_idx, object_ids, masks in predictor.propagate_in_video(
                    inference_state, reverse=reverse
                ):
                    for i, obj_id in enumerate(object_ids):
                        mask_data = (masks[i, 0] > 0.0).cpu().numpy().astype(
                            np.uint8
                        ) * 255
                        obj_frames.setdefault(int(obj_id), {})[frame_idx] = mask_data

        frame_count = int(inference_state["num_frames"])
        expected_object_ids = sorted(
            {int(prompt.object_id) for prompt in prompts.prompts}
        )
        if strict_generation:
            if sorted(obj_frames) != expected_object_ids:
                raise RuntimeError("SAM2 did not produce every prompted object")
            expected_frames = set(range(frame_count))
            for object_id in expected_object_ids:
                if set(obj_frames[object_id]) != expected_frames:
                    raise RuntimeError(
                        f"SAM2 object {object_id} did not produce every video frame"
                    )

        for obj_id, frames_dict in obj_frames.items():
            out = write_root / f"{obj_id}{mask_extension}"
            writer = FrameWriter.from_path(out)
            try:
                for fidx in sorted(frames_dict):
                    writer.write_frame(frames_dict[fidx], stem=f"{fidx:06d}")
            finally:
                writer.close()

        if not strict_generation:
            return None
        current_static_identity = build_static_identity(
            video_path,
            prompts_path,
            weights_dir,
            prompt_data.get("prompts") or [],
            image_id,
        )
        if current_static_identity != static_identity:
            raise RuntimeError(
                "SAM2 inputs changed during inference; refusing the atomic commit"
            )
        commit_generation(write_root, static_identity, expected_object_ids, frame_count)
        validated = validate_generation(write_root, static_identity)
        if os.path.lexists(output):
            raise FileExistsError(
                f"SAM2 output appeared during generation; refusing overwrite: {output}"
            )
        os.replace(write_root, output)
        return validated
    finally:
        if strict_generation and write_root.exists():
            shutil.rmtree(write_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video to masks using SAM2")
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video"
    )
    parser.add_argument(
        "--prompts_path", type=str, required=True, help="Path to prompts JSON file"
    )
    parser.add_argument(
        "--masks_dir", type=str, required=True, help="Output directory for masks"
    )
    parser.add_argument(
        "--weights_dir", type=str, required=True, help="Path to SAM2 weights directory"
    )
    parser.add_argument(
        "--image_id",
        help="Immutable sha256:<64 hex> ID; enables strict atomic generation semantics.",
    )

    args = parser.parse_args()
    video_to_masks(
        args.video_path,
        args.prompts_path,
        args.masks_dir,
        args.weights_dir,
        image_id=args.image_id,
    )
