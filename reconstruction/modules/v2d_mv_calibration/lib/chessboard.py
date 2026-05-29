# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""Chessboard corner detection for multi-camera calibration."""

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from v2d.common.video import FrameSource


logger = logging.getLogger(__name__)


def _chessboard_detect_worker(
    source_paths: list[str],
    board_size: tuple[int, int],
    start_idx: int,
    end_idx: int,
    frames_slice: slice | None = None,
    progress_queue: Any = None,
) -> tuple[list[list[np.ndarray | None]], list[int]]:
    sources = [FrameSource.from_path(p, frames_slice=frames_slice) for p in source_paths]

    chessboard_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FILTER_QUADS
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    correspondences = []
    frame_indices = []

    for t in range(start_idx, end_idx):
        row_t = []
        found = 0
        for src in sources:
            img = src[t]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            ret, corners = cv2.findChessboardCorners(gray, board_size, chessboard_flags)
            if ret:
                corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                row_t.append(np.squeeze(corners_refined))  # (N, 2)
                found += 1
            else:
                row_t.append(None)

        if found >= 2:
            correspondences.append(row_t)
            frame_indices.append(t)

        if progress_queue is not None:
            progress_queue.put(1)

    for src in sources:
        src.close()

    return correspondences, frame_indices


def chessboard_extract_correspondences(
    source_paths: list[str | Path],
    board_size: tuple[int, int] = (9, 6),
    num_workers: int = 8,
    frames_slice: slice | None = None,
) -> tuple[list[list[np.ndarray | None]], list[int]]:
    """Extract chessboard correspondences from multi-camera images.

    Args:
        source_paths: Per-camera paths (directory or .h5) for FrameSource.
        board_size: (width, height) inner corners of the chessboard.
        num_workers: Number of parallel workers.
        frames_slice: Optional slice to limit frame range.

    Returns:
        Tuple of (correspondences, frame_indices).
    """
    src_path_strs = [str(p) for p in source_paths]
    temp_sources = [FrameSource.from_path(p, frames_slice=frames_slice) for p in source_paths]
    per_cam_counts = [s.n_frames for s in temp_sources]
    for s in temp_sources:
        s.close()

    N = len(src_path_strs)
    L = min(per_cam_counts)

    if max(per_cam_counts) != L:
        logger.warning(
            "Camera image counts differ: %s. Clipping to %d frames.",
            per_cam_counts, L,
        )

    logger.info(
        f"Extracting chessboard correspondences"
        f"\n\t- Number of cameras: {N}"
        f"\n\t- Number of frames: {L}"
        f"\n\t- Board size: {board_size}"
        f"\n\t- Number of workers: {num_workers}"
    )

    with multiprocessing.Manager() as manager:
        progress_queue = manager.Queue()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_workers):
                start_idx = (L // num_workers) * i + min(i, L % num_workers)
                end_idx = start_idx + (L // num_workers) + (1 if i < L % num_workers else 0)

                futures.append(
                    executor.submit(
                        _chessboard_detect_worker,
                        src_path_strs,
                        board_size,
                        start_idx,
                        end_idx,
                        frames_slice,
                        progress_queue,
                    )
                )

            with tqdm(total=L, desc="Detecting chessboards") as pbar:
                completed_frames = 0
                while completed_frames < L:
                    if any(f.done() and f.exception() for f in futures):
                        [f.result() for f in futures]
                    try:
                        while not progress_queue.empty():
                            progress_queue.get_nowait()
                            pbar.update(1)
                            completed_frames += 1
                    except Exception:
                        pass

            results = [future.result() for future in futures]
            correspondences = [row for corrs, _ in results for row in corrs]
            frame_indices = [idx for _, idxs in results for idx in idxs]
            return correspondences, frame_indices
