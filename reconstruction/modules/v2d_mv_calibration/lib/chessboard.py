# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


def _detect_chessboard(
    gray: np.ndarray,
    board_size: tuple[int, int],
    use_marker_chessboard: bool = False,
) -> np.ndarray | None:
    """Detect and canonically order one chessboard observation."""
    expected_count = board_size[0] * board_size[1]

    if use_marker_chessboard:
        detector = getattr(cv2, "findChessboardCornersSBWithMeta", None)
        if detector is None:
            raise RuntimeError(
                "Marker chessboard detection requires an OpenCV build with "
                "findChessboardCornersSBWithMeta"
            )

        marker_flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
            | cv2.CALIB_CB_MARKER
        )
        ret, corners, meta = detector(gray, board_size, marker_flags)
        if not ret or corners is None or meta is None:
            return None

        corners = np.asarray(corners)
        meta = np.asarray(meta)
        if corners.size != expected_count * 2:
            logger.debug(
                "Rejecting marker chessboard with %d coordinates; expected %d",
                corners.size,
                expected_count * 2,
            )
            return None

        expected_shape = (board_size[1], board_size[0])
        transposed_shape = (board_size[0], board_size[1])
        if meta.shape == expected_shape:
            corners_grid = corners.reshape(*expected_shape, 2)
            meta_grid = meta
        elif meta.shape == transposed_shape and transposed_shape != expected_shape:
            corners_grid = corners.reshape(*transposed_shape, 2).transpose(1, 0, 2)
            meta_grid = meta.T
        else:
            logger.debug(
                "Rejecting marker chessboard metadata shape %s; expected %s or %s",
                meta.shape,
                expected_shape,
                transposed_shape,
            )
            return None

        if np.count_nonzero(meta_grid == 4) != 1:
            logger.debug("Rejecting marker chessboard without exactly one origin marker")
            return None
        return corners_grid.reshape(expected_count, 2)

    chessboard_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FILTER_QUADS
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    ret, corners = cv2.findChessboardCorners(gray, board_size, chessboard_flags)
    if not ret:
        return None
    corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return np.squeeze(corners_refined)


def _chessboard_detect_worker(
    source_paths: list[str],
    board_size: tuple[int, int],
    start_idx: int,
    end_idx: int,
    frames_slice: slice | None = None,
    progress_queue: Any = None,
    use_marker_chessboard: bool = False,
) -> tuple[list[list[np.ndarray | None]], list[int]]:
    sources = [FrameSource.from_path(p, frames_slice=frames_slice) for p in source_paths]

    correspondences = []
    frame_indices = []

    for t in range(start_idx, end_idx):
        row_t = []
        found = 0
        for src in sources:
            img = src[t]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            corners = _detect_chessboard(
                gray,
                board_size,
                use_marker_chessboard=use_marker_chessboard,
            )
            if corners is not None:
                row_t.append(corners)  # (N, 2)
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
    use_marker_chessboard: bool = False,
) -> tuple[list[list[np.ndarray | None]], list[int]]:
    """Extract chessboard correspondences from multi-camera images.

    Args:
        source_paths: Per-camera paths (directory or .h5) for FrameSource.
        board_size: (width, height) inner corners of the chessboard.
        num_workers: Number of parallel workers.
        frames_slice: Optional slice to limit frame range.
        use_marker_chessboard: Use the marker-aware SB detector for a
            three-dot asymmetric checkerboard. Defaults to the legacy detector.

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
        f"\n\t- Detector: {'marker_sb' if use_marker_chessboard else 'legacy'}"
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
                        use_marker_chessboard,
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
