# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E2FGVI-HQ inference implementation. Heavy imports are intentionally lazy."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .validation import RunPlan, probe_video, select_reference_indices, validate_mask_array


def _read_rgb_frames(path: str | os.PathLike[str], plan: RunPlan) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not reopen input video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (plan.video.height, plan.video.width):
                raise RuntimeError("input video geometry changed after validation")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) != plan.video.frame_count:
        raise RuntimeError(
            f"input video decoded {len(frames)} frames after validation; expected {plan.video.frame_count}"
        )
    return frames


def _resize_frames(frames: list[np.ndarray], width: int, height: int) -> list[np.ndarray]:
    if frames[0].shape[:2] == (height, width):
        return frames
    return [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames]


def _prepare_masks(masks_path: str | os.PathLike[str], plan: RunPlan) -> np.ndarray:
    source_masks = np.load(masks_path, allow_pickle=False)
    validate_mask_array(source_masks, plan.video)
    masks = np.asarray(source_masks, dtype=np.uint8)
    if (plan.processing_width, plan.processing_height) != (
        plan.video.width,
        plan.video.height,
    ):
        masks = np.stack(
            [
                cv2.resize(
                    mask,
                    (plan.processing_width, plan.processing_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                for mask in masks
            ]
        )
    if plan.config.dilation_iterations:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (plan.config.dilation_kernel, plan.config.dilation_kernel),
        )
        masks = np.stack(
            [
                cv2.dilate(
                    mask,
                    kernel,
                    iterations=plan.config.dilation_iterations,
                )
                for mask in masks
            ]
        )
    return masks.astype(bool, copy=False)


def _load_model(checkpoint_path: str | os.PathLike[str], device_name: str, seed: int) -> tuple[Any, Any, Any]:
    # PyTorch requires this to be present before the first CUDA/cuBLAS call for
    # deterministic matrix multiplication.  Without it, deterministic mode
    # only emits warnings and does not provide the reproducibility promised by
    # the run metadata.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({device_name}) but CUDA is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    try:
        from E2FGVI.model.e2fgvi_hq import InpaintGenerator
    except ImportError as exc:
        raise RuntimeError(
            "E2FGVI source is not importable; run inside the supplied container image"
        ) from exc

    model = InpaintGenerator().to(device)
    # Never fall back to unrestricted pickle deserialization.  The published
    # HQ state dictionary is compatible with weights_only=True, and treating a
    # checkpoint as executable Python would make a local model file a code-
    # execution boundary.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint does not contain a model state dictionary")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return torch, device, model


def _pad_for_hq(tensor: Any, height: int, width: int) -> Any:
    """Mirror-pad using the exact concatenation strategy from E2FGVI test.py."""
    import torch

    height_pad = (60 - height % 60) % 60
    width_pad = (108 - width % 108) % 108
    if height_pad:
        target_height = height + height_pad
        while tensor.shape[3] < target_height:
            tensor = torch.cat([tensor, torch.flip(tensor, [3])], dim=3)
        tensor = tensor[:, :, :, :target_height, :]
    if width_pad:
        target_width = width + width_pad
        while tensor.shape[4] < target_width:
            tensor = torch.cat([tensor, torch.flip(tensor, [4])], dim=4)
        tensor = tensor[:, :, :, :, :target_width]
    return tensor


def _write_video_atomic(
    output_path: str | os.PathLike[str],
    frames_rgb: list[np.ndarray],
    fps: float,
    codec: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial-{os.getpid()}{output.suffix}")
    if temporary.exists():
        temporary.unlink()
    height, width = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open output video writer for {temporary} with codec {codec}")
    try:
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    try:
        encoded = probe_video(temporary)
        expected_count = len(frames_rgb)
        if (encoded.width, encoded.height, encoded.frame_count) != (width, height, expected_count):
            raise RuntimeError(
                "encoded output geometry/count mismatch: "
                f"got {encoded.width}x{encoded.height}/{encoded.frame_count}, "
                f"expected {width}x{height}/{expected_count}"
            )
        if not math.isclose(encoded.fps, fps, rel_tol=1e-4, abs_tol=1e-3):
            raise RuntimeError(f"encoded output FPS {encoded.fps} does not preserve source FPS {fps}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_inference(
    plan: RunPlan,
    input_video: str | os.PathLike[str],
    masks_path: str | os.PathLike[str],
    checkpoint_path: str | os.PathLike[str],
    output_video: str | os.PathLike[str],
) -> None:
    """Run E2FGVI and write a video at the source resolution and FPS."""
    torch, device, model = _load_model(checkpoint_path, plan.config.device, plan.config.seed)
    source_frames = _read_rgb_frames(input_video, plan)
    processing_frames = _resize_frames(
        source_frames,
        plan.processing_width,
        plan.processing_height,
    )
    masks = _prepare_masks(masks_path, plan)
    height, width = plan.processing_height, plan.processing_width
    completed: list[np.ndarray | None] = [None] * plan.video.frame_count
    blend_counts = [0] * plan.video.frame_count

    with torch.inference_mode():
        for center in range(0, plan.video.frame_count, plan.config.neighbor_stride):
            neighbors = list(
                range(
                    max(0, center - plan.config.neighbor_stride),
                    min(plan.video.frame_count, center + plan.config.neighbor_stride + 1),
                )
            )
            references = select_reference_indices(
                center,
                neighbors,
                plan.video.frame_count,
                plan.config.ref_stride,
                plan.config.num_ref,
            )
            selected = neighbors + references
            frame_batch = np.stack([processing_frames[index] for index in selected])
            mask_batch = np.stack([masks[index] for index in selected])
            images = (
                torch.from_numpy(frame_batch)
                .permute(0, 3, 1, 2)
                .to(device=device, dtype=torch.float32)
                .div_(127.5)
                .sub_(1.0)
                .unsqueeze(0)
            )
            mask_tensor = (
                torch.from_numpy(mask_batch)
                .to(device=device, dtype=torch.float32)
                .unsqueeze(1)
                .unsqueeze(0)
            )
            masked_images = images * (1.0 - mask_tensor)
            masked_images = _pad_for_hq(masked_images, height, width)
            predictions, predicted_flows = model(masked_images, len(neighbors))
            del predicted_flows
            predictions = predictions[: len(neighbors), :, :height, :width]
            predictions = (
                predictions.add(1.0)
                .div(2.0)
                .clamp_(0.0, 1.0)
                .mul(255.0)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            del images, mask_tensor, masked_images

            for local_index, frame_index in enumerate(neighbors):
                mask = masks[frame_index, :, :, None]
                candidate = np.where(mask, predictions[local_index], processing_frames[frame_index])
                count = blend_counts[frame_index]
                if completed[frame_index] is None:
                    completed[frame_index] = candidate
                else:
                    completed[frame_index] = cv2.addWeighted(
                        completed[frame_index],
                        count / (count + 1),
                        candidate,
                        1 / (count + 1),
                        0,
                    )
                blend_counts[frame_index] += 1

    if any(frame is None for frame in completed):
        missing = [index for index, frame in enumerate(completed) if frame is None]
        raise RuntimeError(f"E2FGVI did not produce frames: {missing}")

    output_frames: list[np.ndarray] = []
    for index, inpainted in enumerate(completed):
        assert inpainted is not None
        if (width, height) != (plan.video.width, plan.video.height):
            inpainted = cv2.resize(
                inpainted,
                (plan.video.width, plan.video.height),
                interpolation=cv2.INTER_CUBIC,
            )
            output_mask = cv2.resize(
                masks[index].astype(np.uint8),
                (plan.video.width, plan.video.height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            output_mask = masks[index]
        output_frames.append(np.where(output_mask[:, :, None], inpainted, source_frames[index]))

    _write_video_atomic(
        output_video,
        output_frames,
        plan.video.fps,
        plan.config.codec,
    )
