"""Chessboard corner detection for multi-camera calibration."""

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)


def _read_png(path: Path) -> np.ndarray:
    return iio.imread(path, plugin="pillow")


def _chessboard_detect_worker(
    image_file_lists: list[list[Path]],
    board_size: tuple[int, int],
    start_idx: int,
    end_idx: int,
    progress_queue: Any = None,
) -> list[list[np.ndarray | None]]:
    chessboard_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FILTER_QUADS
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    correspondences = []

    for t in range(start_idx, end_idx):
        row_t = []
        found = 0
        for cam_files in image_file_lists:
            img = _read_png(cam_files[t])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, board_size, chessboard_flags)
            if ret:
                corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                row_t.append(np.squeeze(corners_refined))  # (N, 2)
                found += 1
            else:
                row_t.append(None)

        if found >= 2:
            correspondences.append(row_t)

        if progress_queue is not None:
            progress_queue.put(1)

    return correspondences


def chessboard_extract_correspondences(
    image_file_lists: list[list[Path]],
    board_size: tuple[int, int],
    num_workers: int = 8,
) -> list[list[np.ndarray | None]]:
    """Extract chessboard correspondences from multi-camera images.

    Args:
        image_file_lists: List of per-camera file lists (sorted PNG paths).
        board_size: (width, height) inner corners of the chessboard.
        num_workers: Number of parallel workers.

    Returns:
        List of frames, each a list of per-camera correspondences
        (either (P, 2) array or None if not detected).
    """
    N = len(image_file_lists)
    L = min(len(f) for f in image_file_lists)

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
                        image_file_lists,
                        board_size,
                        start_idx,
                        end_idx,
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

            correspondences = [future.result() for future in futures]
            return [row for per_worker in correspondences for row in per_worker]
