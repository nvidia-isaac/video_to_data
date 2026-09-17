# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack SAM2 human and object mask streams into the CARI4D wild-video schema."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from v2d.common.video import FrameSource


MASK_SCHEMA = "v2d.cari4d.wild_masks.v1"


def _sequence_name(video_path: Path) -> str:
    suffix = ".0.color.mp4"
    if not video_path.name.endswith(suffix):
        raise ValueError(f"CARI4D input video must end with {suffix}: {video_path}")
    return video_path.name[:-len(suffix)]


def _mask(mask: np.ndarray, expected_shape: tuple[int, int], role: str, stem: str) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape != expected_shape:
        raise ValueError(f"{role} mask {stem} has shape {mask.shape}, expected {expected_shape}")
    if not np.issubdtype(mask.dtype, np.bool_) and not np.issubdtype(mask.dtype, np.integer):
        raise TypeError(f"{role} mask {stem} must have boolean or integer dtype, got {mask.dtype}")
    return (mask > 0).astype(np.uint8) * 255


def masks_pack_cari4d_h5(video_path: str | Path, human_masks_path: str | Path, object_masks_path: str | Path, output_path: str | Path, *, overwrite: bool = False) -> Path:
    video_path, human_masks_path, object_masks_path, output_path = map(lambda value: Path(value).resolve(), (video_path, human_masks_path, object_masks_path, output_path))
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    for path in (human_masks_path, object_masks_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    sequence = _sequence_name(video_path)
    with FrameSource.from_path(video_path) as video, FrameSource.from_path(human_masks_path) as human_masks, FrameSource.from_path(object_masks_path) as object_masks:
        if video.n_frames <= 0:
            raise ValueError(f"CARI4D input video contains no frames: {video_path}")
        expected_stems = [f"{index:06d}" for index in range(video.n_frames)]
        for role, source in (("human", human_masks), ("object", object_masks)):
            if source.n_frames != video.n_frames:
                raise ValueError(f"{role} mask count {source.n_frames} differs from video frame count {video.n_frames}")
            if source.image_size != video.image_size:
                raise ValueError(f"{role} mask size {source.image_size} differs from video size {video.image_size}")
            if source.stems != expected_stems:
                raise ValueError(f"{role} mask stems must exactly cover 000000 through {expected_stems[-1]}")
        width, height = video.image_size
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        with h5py.File(temporary, "w") as output:
            output.attrs["schema"] = MASK_SCHEMA
            output.attrs["sequence"] = sequence
            output.attrs["frame_count"] = video.n_frames
            output.attrs["width"] = width
            output.attrs["height"] = height
            group = output.require_group(sequence)
            for stem, human, obj in zip(expected_stems, human_masks.iter_frames(), object_masks.iter_frames(), strict=True):
                group.create_dataset(f"{stem}-k0.person_mask.png", data=_mask(human, (height, width), "human", stem), compression="lzf", shuffle=True)
                group.create_dataset(f"{stem}-k0.obj_rend_mask.png", data=_mask(obj, (height, width), "object", stem), compression="lzf", shuffle=True)
            output.flush()
        os.replace(temporary, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack SAM2 human and object masks into a CARI4D H5 file")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--human_masks_path", required=True)
    parser.add_argument("--object_masks_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(masks_pack_cari4d_h5(**vars(args)))


if __name__ == "__main__":
    main()
