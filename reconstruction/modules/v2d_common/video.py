# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frame I/O utilities: FrameSource (read), FrameWriter (write), and video helpers.

Moved here from ``v2d.mv.io.video`` so that all modules can use frame I/O
via ``v2d-common[io]`` without pulling in ``v2d-mv``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import av
import cv2
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm

try:
    from .hdf5_transcode import decode_rgb_jpeg
    from .ffv1_sidecar import (
        is_ffv1_sidecar_h5,
        read_ffv1_metadata,
        verify_ffv1_sidecar,
    )
except ImportError:  # Support direct module imports in lightweight tests.
    from hdf5_transcode import decode_rgb_jpeg
    from ffv1_sidecar import (
        is_ffv1_sidecar_h5,
        read_ffv1_metadata,
        verify_ffv1_sidecar,
    )

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def get_video_lwh(video_path: Path) -> tuple[int, int, int]:
    L, H, W, _ = iio.improps(video_path, plugin="pyav").shape
    if L <= 0 or not np.isfinite(L):
        # Fragmented MP4 and other containers may omit the stream frame count.
        # Decode incrementally without retaining frames or converting to RGB.
        with av.open(str(video_path)) as container:
            L = sum(1 for _ in container.decode(video=0))
    return L, W, H


def get_video_reader(video_path: Path) -> Iterator[np.ndarray]:
    return iio.imiter(video_path, plugin="pyav")


def get_video_writer(video_path: Path, fps: int = 30, crf: int = 17) -> Any:
    """Return an imageio video writer (H.264).  Remember to call ``.close()``.

    Args:
        video_path: Destination file.
        fps: Frames per second.
        crf: 0 = lossless, 17 = visually lossless, 23 = default.
    """
    writer = iio.imopen(video_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)
    stream = writer._video_stream

    if stream.codec_context.time_base is None:
        stream.codec_context.time_base = Fraction(1, fps)

    stream.options = {"crf": str(crf)}
    return writer


def read_video_np(
    video_path: Path,
    start_frame: int = 0,
    end_frame: int = -1,
    scale: float = 1.0,
) -> np.ndarray:
    """Read a video (or a sub-range) into a ``(N, H, W, 3)`` uint8 array."""
    filter_args = []
    should_check_length = False

    if not (start_frame == 0 and end_frame == -1):
        if end_frame == -1:
            filter_args.append(("trim", f"start_frame={start_frame}"))
        else:
            should_check_length = True
            filter_args.append(
                ("trim", f"start_frame={start_frame}:end_frame={end_frame}")
            )

    if scale != 1.0:
        filter_args.append(("scale", f"iw*{scale}:ih*{scale}"))

    frames = iio.imread(video_path, plugin="pyav", filter_sequence=filter_args)
    if should_check_length:
        assert len(frames) == end_frame - start_frame

    return frames


def save_video_np(images: np.ndarray, video_path: Path, fps: int = 30, crf: int = 17):
    images = np.array(images).astype(np.uint8)
    with get_video_writer(video_path, fps=fps, crf=crf) as writer:
        writer.write(images)


def generate_video_from_image_dir(
    input_dir: Path,
    output_file: Path,
    fps: int = 30,
    crf: int = 17,
):
    """Running ffmpeg as a subprocess is much faster than imageio."""
    subprocess.run([
        "ffmpeg",
        "-framerate", str(fps),
        "-i", input_dir / "%06d.png",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        output_file,
        "-y",
    ])


def pack_directory_to_h5(
    image_dir: Path | str,
    h5_path: Path | str,
    remove_pngs: bool = False,
    show_progress: bool = True,
    *,
    compression: str | int | None = "gzip",
    compression_opts: Any = 1,
    shuffle: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
    reindex_stems: bool = False,
) -> None:
    """Pack a directory of sorted PNGs into a single HDF5 file.

    Used by rosbag_to_edex after synchronization -- the only module that
    must write PNGs first (for sync), then pack.
    """
    import h5py

    image_dir = Path(image_dir)
    h5_path = Path(h5_path)
    all_png_files = sorted(image_dir.glob("*.png"))
    if not all_png_files:
        raise FileNotFoundError(f"No PNGs to pack in {image_dir}")
    start = int(start_frame)
    end = len(all_png_files) if end_frame is None else int(end_frame)
    if start < 0 or end <= start or end > len(all_png_files):
        raise ValueError(
            f"Invalid PNG frame range [{start}, {end}) for "
            f"{len(all_png_files)} frames in {image_dir}"
        )
    png_files = all_png_files[start:end]

    stems = (
        [f"{index:06d}" for index in range(len(png_files))]
        if reindex_stems
        else [p.stem for p in png_files]
    )
    first = iio.imread(png_files[0])
    h, w = first.shape[:2]
    frame_shape = first.shape

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        ds = f.create_dataset(
            "frames",
            shape=(len(png_files), *frame_shape),
            dtype=first.dtype,
            chunks=(1, *frame_shape),
            compression=compression,
            compression_opts=compression_opts,
            shuffle=shuffle,
        )
        it = tqdm(png_files, desc=f"Packing {image_dir.name}") if show_progress else png_files
        for i, p in enumerate(it):
            ds[i] = iio.imread(p)

        f.attrs["stems"] = json.dumps(stems)
        f.attrs["n_frames"] = len(png_files)
        f.attrs["width"] = w
        f.attrs["height"] = h

    if remove_pngs:
        import shutil
        for p in all_png_files:
            p.unlink()
        shutil.rmtree(image_dir)

    logger.info("Packed %d frames from %s -> %s", len(png_files), image_dir, h5_path)


# ---------------------------------------------------------------------------
# FrameSource (base) + backend subclasses
# ---------------------------------------------------------------------------

class FrameSource:
    """Read-only frame sequence backed by an image directory, video, or HDF5.

    Construct via the :meth:`from_path` factory, which auto-detects the
    backend from the path.  All backends expose the same public interface:

    * ``n_frames``, ``image_size``, ``stems``, ``path``
    * ``__getitem__(i)`` -- random access (not available for video)
    * ``iter_frames()`` / ``iter_batches(batch_size)`` -- sequential
    * ``close()`` / context-manager protocol
    """

    n_frames: int
    image_size: tuple[int, int]  # (W, H)
    _path: Path
    _stems: list[str]

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------

    @classmethod
    def from_path(cls, path: str | Path, frames_slice: slice | None = None) -> "FrameSource":
        """Auto-detect backend from *path*.

        Resolution order:
          1. missing ``camera.h5`` with sibling ``camera/`` -> legacy image dir
          2. ``.h5`` / ``.hdf5`` suffix  ->  :class:`_HDF5Source`
          3. existing directory           ->  :class:`_ImageDirSource`
          4. video extension              ->  :class:`_VideoSource`
        """
        path = Path(path)
        if path.suffix in (".h5", ".hdf5") and not path.exists():
            legacy_dir = path.with_suffix("")
            if legacy_dir.is_dir():
                path = legacy_dir
        if path.suffix in (".h5", ".hdf5"):
            if is_ffv1_sidecar_h5(path):
                return _FFV1SidecarSource(path, frames_slice)
            return _HDF5Source(path, frames_slice)
        if path.is_dir():
            return _ImageDirSource(path, frames_slice)
        if path.suffix.lower() in _VIDEO_EXTENSIONS:
            return _VideoSource(path, frames_slice)
        raise ValueError(f"Cannot auto-detect FrameSource backend for: {path}")

    # ------------------------------------------------------------------
    # shared properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Original path passed at construction."""
        return self._path

    @property
    def stems(self) -> list[str]:
        """Frame stem strings (e.g. ``['000000', '000001', ...]``)."""
        return self._stems

    @property
    def image_paths(self) -> list[Path]:
        """Per-frame file paths (only for image-directory backend)."""
        raise RuntimeError("image_paths is only available for image-directory sources")

    @property
    def has_file_paths(self) -> bool:
        """True when individual image file paths are available."""
        return False

    # ------------------------------------------------------------------
    # read interface (subclasses override)
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.n_frames

    def iter_frames(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def iter_batches(self, batch_size: int):
        """Yield ``(batch_start_index, list[np.ndarray])`` tuples."""
        batch: list[np.ndarray] = []
        batch_start = 0
        for frame in self.iter_frames():
            batch.append(frame)
            if len(batch) == batch_size:
                yield batch_start, batch
                batch_start += len(batch)
                batch = []
        if batch:
            yield batch_start, batch

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _ImageDirSource(FrameSource):
    """Reads sorted PNG/JPG files from a directory."""

    def __init__(self, image_dir: Path | str, frames_slice: slice | None = None):
        image_dir = Path(image_dir)
        self._path = image_dir
        paths: list[Path] = []
        for ext in ("png", "jpg", "jpeg"):
            paths = sorted(image_dir.glob(f"*.{ext}"))
            if paths:
                break
        if not paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        if frames_slice is not None:
            paths = paths[frames_slice]
        self._image_paths_list = paths
        self._stems = [p.stem for p in paths]
        first = iio.imread(paths[0])
        self.n_frames = len(paths)
        self.image_size = (first.shape[1], first.shape[0])

    @property
    def image_paths(self) -> list[Path]:
        return self._image_paths_list

    @property
    def has_file_paths(self) -> bool:
        return True

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0:
            idx += self.n_frames
        if idx < 0 or idx >= self.n_frames:
            raise IndexError(f"Frame index {idx} out of range [0, {self.n_frames})")
        return iio.imread(self._image_paths_list[idx])

    def iter_frames(self) -> Iterator[np.ndarray]:
        for p in self._image_paths_list:
            yield iio.imread(p)

    def iter_batches(self, batch_size: int):
        for i in range(0, self.n_frames, batch_size):
            batch_paths = self._image_paths_list[i : i + batch_size]
            yield i, [iio.imread(p) for p in batch_paths]


class _HDF5Source(FrameSource):
    """Reads frames from an HDF5 file with a ``frames`` dataset."""

    def __init__(self, h5_path: Path | str, frames_slice: slice | None = None):
        h5_path = Path(h5_path)
        self._path = h5_path
        self._h5_path = h5_path
        self._h5_file = None
        self._h5_dataset = None
        self._h5_indices: list[int] | None = None
        self._frame_encoding: str | None = None
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        self._open_h5()
        ds = self._h5_dataset
        total = ds.shape[0]
        frame_encoding = self._h5_file.attrs.get("frame_encoding")
        if isinstance(frame_encoding, bytes):
            frame_encoding = frame_encoding.decode("utf-8")
        self._frame_encoding = frame_encoding
        if self._frame_encoding == "jpeg":
            if ds.ndim != 1:
                raise ValueError("JPEG HDF5 frames must be a one-dimensional dataset")
            # The on-disk dataset contains variable-length encoded byte
            # payloads, but FrameSource exposes the decoded RGB frames.
            self.dtype = np.dtype(np.uint8)
            self.image_size = (
                int(self._h5_file.attrs["width"]),
                int(self._h5_file.attrs["height"]),
            )
        elif self._frame_encoding is not None:
            raise ValueError(f"Unsupported HDF5 frame encoding: {self._frame_encoding}")
        elif ds.ndim >= 3:
            self.dtype = np.dtype(ds.dtype)
            self.image_size = (ds.shape[2], ds.shape[1])
        else:
            raise ValueError(
                "One-dimensional HDF5 frame datasets require a supported "
                "frame_encoding attribute"
            )
        stems_json = self._h5_file.attrs.get("stems")
        all_stems = json.loads(stems_json) if stems_json is not None else [f"{i:06d}" for i in range(total)]

        if frames_slice is not None:
            self._h5_indices = list(range(total))[frames_slice]
            self._stems = [all_stems[i] for i in self._h5_indices]
            self.n_frames = len(self._h5_indices)
        else:
            self._stems = all_stems
            self.n_frames = total

    def _open_h5(self):
        if self._h5_file is None:
            import h5py
            self._h5_file = h5py.File(self._h5_path, "r")
            self._h5_dataset = self._h5_file["frames"]

    def _physical(self, idx: int) -> int:
        return self._h5_indices[idx] if self._h5_indices is not None else idx

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0:
            idx += self.n_frames
        if idx < 0 or idx >= self.n_frames:
            raise IndexError(f"Frame index {idx} out of range [0, {self.n_frames})")
        self._open_h5()
        frame = self._h5_dataset[self._physical(idx)]
        return decode_rgb_jpeg(frame) if self._frame_encoding == "jpeg" else frame

    def iter_frames(self) -> Iterator[np.ndarray]:
        self._open_h5()
        for i in range(self.n_frames):
            frame = self._h5_dataset[self._physical(i)]
            yield decode_rgb_jpeg(frame) if self._frame_encoding == "jpeg" else frame

    def iter_batches(self, batch_size: int):
        self._open_h5()
        for i in range(0, self.n_frames, batch_size):
            end = min(i + batch_size, self.n_frames)
            frames = [self._h5_dataset[self._physical(j)] for j in range(i, end)]
            if self._frame_encoding == "jpeg":
                frames = [decode_rgb_jpeg(frame) for frame in frames]
            yield i, frames

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._h5_dataset = None


class _FFV1SidecarSource(FrameSource):
    """Reads exact frames from an FFV1 Matroska sidecar committed by HDF5."""

    def __init__(self, metadata_path: Path | str, frames_slice: slice | None = None):
        metadata_path = Path(metadata_path)
        self._path = metadata_path
        self._metadata = read_ffv1_metadata(metadata_path)
        self._video_path = self._metadata["sidecar_path"]
        self.image_size = (self._metadata["width"], self._metadata["height"])
        total = self._metadata["n_frames"]
        self._all_stems = self._metadata["stems"]
        self._all_pts = self._metadata["frame_pts"]
        self._pts_to_index = {pts: index for index, pts in enumerate(self._all_pts)}
        self._keyframes = self._metadata["keyframe_indices"]
        if frames_slice is None:
            self._indices = list(range(total))
        else:
            self._indices = list(range(total))[frames_slice]
        self._stems = [self._all_stems[index] for index in self._indices]
        self.n_frames = len(self._indices)

    def _format(self) -> str:
        return "rgb24" if self._metadata["kind"] == "rgb" else "gray16le"

    def _physical(self, idx: int) -> int:
        return self._indices[idx]

    def __getitem__(self, idx: int) -> np.ndarray:
        if idx < 0:
            idx += self.n_frames
        if idx < 0 or idx >= self.n_frames:
            raise IndexError(f"Frame index {idx} out of range [0, {self.n_frames})")
        target = self._physical(idx)
        keyframe = max(index for index in self._keyframes if index <= target)
        with av.open(str(self._video_path)) as container:
            stream = container.streams.video[0]
            container.seek(
                self._all_pts[keyframe],
                stream=stream,
                backward=True,
                any_frame=False,
            )
            for frame in container.decode(stream):
                physical = self._pts_to_index.get(int(frame.pts))
                if physical == target:
                    return frame.to_ndarray(format=self._format())
                if physical is not None and physical > target:
                    break
        raise RuntimeError(f"Unable to seek to FFV1 frame {target}")

    def iter_frames(self) -> Iterator[np.ndarray]:
        # Normal forward slices stream in constant memory. Reordered/repeated
        # slices are uncommon, but preserve FrameSource semantics via indexing.
        if self._indices == sorted(set(self._indices)):
            wanted = iter(self._indices)
            target = next(wanted, None)
            with av.open(str(self._video_path)) as container:
                for physical, frame in enumerate(container.decode(video=0)):
                    if target is None:
                        break
                    if physical == target:
                        yield frame.to_ndarray(format=self._format())
                        target = next(wanted, None)
            if target is not None:
                raise RuntimeError(f"FFV1 sidecar is missing frame {target}")
            return
        for index in range(self.n_frames):
            yield self[index]

    def iter_batches(self, batch_size: int):
        batch: list[np.ndarray] = []
        batch_start = 0
        for frame in self.iter_frames():
            batch.append(frame)
            if len(batch) == batch_size:
                yield batch_start, batch
                batch_start += len(batch)
                batch = []
        if batch:
            yield batch_start, batch

    def verify_integrity(self, *, verify_decoded_frames: bool = False) -> dict[str, Any]:
        return verify_ffv1_sidecar(
            self._path, verify_decoded_frames=verify_decoded_frames
        )

    def close(self) -> None:
        # Containers are scoped per operation so random access remains thread-safe.
        return None


class _VideoSource(FrameSource):
    """Sequential-only reader backed by a video file (via pyav)."""

    def __init__(self, video_path: Path | str, frames_slice: slice | None = None):
        video_path = Path(video_path)
        self._path = video_path
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        n, w, h = get_video_lwh(video_path)
        self.n_frames = n
        self.image_size = (w, h)
        self._stems = [f"{i:06d}" for i in range(n)]
        self._video_path = video_path

    def __getitem__(self, idx: int) -> np.ndarray:
        raise RuntimeError(
            "Random access is not supported for video sources. "
            "Use iter_frames() instead."
        )

    def iter_frames(self) -> Iterator[np.ndarray]:
        yield from get_video_reader(self._video_path)

    def iter_batches(self, batch_size: int):
        batch: list[np.ndarray] = []
        batch_start = 0
        for frame in get_video_reader(self._video_path):
            batch.append(frame)
            if len(batch) == batch_size:
                yield batch_start, batch
                batch_start += len(batch)
                batch = []
        if batch:
            yield batch_start, batch


# ---------------------------------------------------------------------------
# FrameWriter (base) + backend subclasses
# ---------------------------------------------------------------------------

class FrameWriter:
    """Write frames to a directory of PNGs, an HDF5 file, or an MP4 video.

    Construct via :meth:`from_path`, which auto-detects the backend from
    the path extension:

      ``.h5``  -> HDF5 (lossless gzip, any dtype)
      ``.mp4`` -> MP4 video (H.264, uint8 RGB only)
      other    -> PNG directory
    """

    _path: Path
    _closed: bool = False

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        fps: int = 30,
        crf: int = 17,
        png_num_workers: int | None = None,
        png_max_pending: int | None = None,
        compression: str | int | None = "gzip",
        compression_opts: Any = 1,
        shuffle: bool = False,
    ) -> "FrameWriter":
        """Auto-detect the backend and return the appropriate writer.

        ``compression``, ``compression_opts``, and ``shuffle`` use h5py's
        dataset terminology and apply only to HDF5 outputs. Non-HDF5 outputs
        reject non-default HDF5 settings so configuration mistakes are visible.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in (".h5", ".hdf5"):
            return _HDF5Writer(
                path,
                compression=compression,
                compression_opts=compression_opts,
                shuffle=shuffle,
            )
        if (compression, compression_opts, shuffle) != ("gzip", 1, False):
            raise ValueError(
                "HDF5 compression options can only be used with .h5/.hdf5 outputs"
            )
        if suffix in _VIDEO_EXTENSIONS:
            return _VideoWriter(path, fps=fps, crf=crf)
        return _PNGDirWriter(
            path,
            num_workers=png_num_workers,
            max_pending=png_max_pending,
        )

    # ------------------------------------------------------------------
    # write interface (subclasses override)
    # ------------------------------------------------------------------

    def write_frame(self, frame: np.ndarray, stem: str | None = None) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


class _PNGDirWriter(FrameWriter):
    """Writes individual PNG files into a directory using background workers."""

    def __init__(
        self,
        path: Path,
        num_workers: int | None = None,
        max_pending: int | None = None,
    ):
        self._path = path
        self._path.mkdir(parents=True, exist_ok=True)
        self._closed = False
        self._png_counter = 0
        if num_workers is None:
            num_workers = min(8, os.cpu_count() or 1)
        self._num_workers = max(1, int(num_workers))
        if max_pending is None:
            max_pending = self._num_workers * 2
        self._max_pending = max(1, int(max_pending))
        self._executor = ThreadPoolExecutor(
            max_workers=self._num_workers,
            thread_name_prefix=f"png-writer-{self._path.name or 'frames'}",
        )
        self._pending_slots = threading.BoundedSemaphore(self._max_pending)
        self._lock = threading.Lock()
        self._futures: set[Future] = set()
        self._errors: list[BaseException] = []

    @staticmethod
    def _write_png(path: Path, frame: np.ndarray) -> None:
        iio.imwrite(path, frame)

    def _raise_if_error(self) -> None:
        with self._lock:
            if self._errors:
                raise self._errors[0]

    def _on_done(self, fut: Future) -> None:
        try:
            fut.result()
        except BaseException as exc:
            with self._lock:
                self._errors.append(exc)
        finally:
            with self._lock:
                self._futures.discard(fut)
            self._pending_slots.release()

    def write_frame(self, frame: np.ndarray, stem: str | None = None) -> None:
        if np.issubdtype(frame.dtype, np.floating):
            raise TypeError(
                f"PNG mode does not accept float arrays (got {frame.dtype}). "
                "Encode to uint8/uint16 first."
            )

        with self._lock:
            if self._closed:
                raise RuntimeError("FrameWriter is closed")
            if self._errors:
                raise self._errors[0]
            if stem is None:
                stem = f"{self._png_counter:06d}"
            self._png_counter += 1
            output_path = self._path / f"{stem}.png"

        self._pending_slots.acquire()
        try:
            self._raise_if_error()
            frame_snapshot = np.array(frame, copy=True, order="C")
            fut = self._executor.submit(self._write_png, output_path, frame_snapshot)
            with self._lock:
                self._futures.add(fut)
            fut.add_done_callback(self._on_done)
        except BaseException:
            self._pending_slots.release()
            raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = list(self._futures)

        error: BaseException | None = None
        for fut in futures:
            try:
                fut.result()
            except BaseException as exc:
                if error is None:
                    error = exc

        self._executor.shutdown(wait=True)
        with self._lock:
            if error is None and self._errors:
                error = self._errors[0]
        if error is not None:
            raise error

    def __del__(self):
        try:
            self.close()
        except BaseException:
            pass


class _HDF5Writer(FrameWriter):
    """Writes frames into a chunked, losslessly compressed HDF5 dataset."""

    def __init__(
        self,
        path: Path,
        *,
        compression: str | int | None = "gzip",
        compression_opts: Any = 1,
        shuffle: bool = False,
    ):
        self._path = path
        self._compression = compression
        self._compression_opts = compression_opts
        self._shuffle = shuffle
        self._closed = False
        self._h5_file = None
        self._h5_dataset = None
        self._h5_stems: list[str] = []
        self._h5_count = 0

    def write_frame(self, frame: np.ndarray, stem: str | None = None) -> None:
        if self._closed:
            raise RuntimeError("FrameWriter is closed")
        import h5py

        if self._h5_file is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._h5_file = h5py.File(self._path, "w")
            shape_tail = frame.shape
            self._h5_dataset = self._h5_file.create_dataset(
                "frames",
                shape=(0, *shape_tail),
                maxshape=(None, *shape_tail),
                dtype=frame.dtype,
                chunks=(1, *shape_tail),
                compression=self._compression,
                compression_opts=self._compression_opts,
                shuffle=self._shuffle,
            )
        ds = self._h5_dataset
        ds.resize(self._h5_count + 1, axis=0)
        ds[self._h5_count] = frame
        self._h5_stems.append(stem if stem is not None else f"{self._h5_count:06d}")
        self._h5_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._h5_file is not None:
            f = self._h5_file
            ds = self._h5_dataset
            f.attrs["stems"] = json.dumps(self._h5_stems)
            f.attrs["n_frames"] = self._h5_count
            if ds is not None and self._h5_count > 0:
                f.attrs["height"] = int(ds.shape[1])
                f.attrs["width"] = int(ds.shape[2]) if ds.ndim >= 3 else 0
            f.close()
            self._h5_file = None
            self._h5_dataset = None


class _VideoWriter(FrameWriter):
    """Writes uint8 RGB frames into an MP4 video via H.264."""

    def __init__(self, path: Path, *, fps: int = 30, crf: int = 17):
        self._path = path
        self._closed = False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = get_video_writer(self._path, fps=fps, crf=crf)

    def write_frame(self, frame: np.ndarray, stem: str | None = None) -> None:
        if self._closed:
            raise RuntimeError("FrameWriter is closed")
        if frame.dtype != np.uint8:
            raise TypeError(
                f"MP4 mode requires uint8 frames (got {frame.dtype})."
            )
        self._writer.write_frame(frame)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._writer.close()


# ---------------------------------------------------------------------------
# tile_videos
# ---------------------------------------------------------------------------

def tile_videos(
    sources: list[Path | FrameSource],
    output_path: Path,
    tile_shape: tuple[int, int],
    output_image_size: tuple[int, int] | None = None,
    video_names: list[str] | None = None,
    show_frame_count: bool = False,
):
    if len(sources) > tile_shape[0] * tile_shape[1]:
        raise ValueError(f"Too many sources to tile: {len(sources)} > {tile_shape[0] * tile_shape[1]}")

    frame_sources = [
        s if isinstance(s, FrameSource) else FrameSource.from_path(s)
        for s in sources
    ]

    frame_counts = [fs.n_frames for fs in frame_sources]
    if len(set(frame_counts)) > 1:
        raise ValueError(f"tile_videos sources have unequal frame counts: {frame_counts}")
    L = frame_counts[0]
    if output_image_size is not None:
        W, H = output_image_size
    else:
        W, H = frame_sources[0].image_size

    W_frame = W * tile_shape[1]
    H_frame = H * tile_shape[0]

    writer = get_video_writer(output_path, fps=30, crf=17)
    iterators = [fs.iter_frames() for fs in frame_sources]
    for l in tqdm(range(L), desc="Tiling videos"):
        img = np.zeros((H_frame, W_frame, 3), dtype=np.uint8)
        for i, it in enumerate(iterators):
            img_tile = next(it)
            if video_names is not None:
                (tw, th), _ = cv2.getTextSize(video_names[i], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(img_tile, video_names[i], (10, th + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if img_tile.shape[:2] != (H, W):
                img_tile = cv2.resize(img_tile, (W, H), interpolation=cv2.INTER_AREA)
            r = i // tile_shape[1]
            c = i % tile_shape[1]
            img[r * H: (r + 1) * H, c * W: (c + 1) * W] = img_tile
        if show_frame_count:
            frame_text = f"Frame {l}"
            (tw, th), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(img, frame_text, (W_frame - tw - 10, th + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        writer.write_frame(img)
    writer.close()
