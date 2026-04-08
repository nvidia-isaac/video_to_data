from pathlib import Path
import subprocess
from typing import Iterator, Any

import cv2
from fractions import Fraction
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm


def get_video_lwh(video_path: Path) -> tuple[int, int, int]:
    L, H, W, _ = iio.improps(video_path, plugin="pyav").shape
    return L, W, H


def get_video_reader(video_path: Path) -> Iterator[np.ndarray]:
    return iio.imiter(video_path, plugin="pyav")


def get_video_writer(video_path: Path, fps: int = 30, crf: int = 17) -> Any:
    """
    Remember to call .close() after writing.
    Args:
        video_path: Path
        fps: int
        crf: 0 is lossless, 17 is visually lossless, 23 is default, +6 results in half the bitrate
    https://trac.ffmpeg.org/wiki/Encode/H.264#crf
    Returns:
        writer: Any
    """
    writer = iio.imopen(video_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)
    stream = writer._video_stream

    # Ensure time_base is valid
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
    """
    Args:
        video_path: Path
        start_frame: int
        end_frame: int
        scale: float
    Returns:
        frames: np.array, (N, H, W, 3) RGB, uint8
    """
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


class FrameSource:
    """Lazy frame source backed by either an image directory or a video file."""

    def __init__(
        self,
        image_dir: Path | None = None,
        video_path: Path | None = None,
        frames_slice: slice | None = None,
    ):
        if (image_dir is None) == (video_path is None):
            raise ValueError("Provide exactly one of image_dir or video_path")

        if image_dir is not None:
            self._image_paths = sorted(Path(image_dir).glob("*.png"))
            if not self._image_paths:
                raise FileNotFoundError(f"No PNG images found in {image_dir}")
            if frames_slice is not None:
                self._image_paths = self._image_paths[frames_slice]
            self._video_path = None
            first = iio.imread(self._image_paths[0])
            self.n_frames = len(self._image_paths)
        else:
            self._image_paths = None
            self._video_path = Path(video_path)
            if not self._video_path.exists():
                raise FileNotFoundError(f"Video file not found: {self._video_path}")
            n, w, h = get_video_lwh(self._video_path)
            first = next(get_video_reader(self._video_path))
            self.n_frames = n

        self.image_size = (first.shape[1], first.shape[0])  # (width, height)

    @property
    def image_paths(self) -> list[Path]:
        if self._image_paths is None:
            raise RuntimeError("image_paths is not available for video-backed FrameSource")
        return self._image_paths

    def iter_batches(self, batch_size: int):
        """Yield ``(batch_start_index, list[np.ndarray])`` tuples."""
        if self._image_paths is not None:
            for i in range(0, self.n_frames, batch_size):
                batch_paths = self._image_paths[i : i + batch_size]
                yield i, [iio.imread(p) for p in batch_paths]
        else:
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

    def iter_frames(self):
        """Yield frames one at a time."""
        if self._image_paths is not None:
            for p in self._image_paths:
                yield iio.imread(p)
        else:
            yield from get_video_reader(self._video_path)


def tile_videos(
    sources: list[Path | FrameSource],
    output_path: Path,
    tile_shape: tuple[int, int],
    output_image_size: tuple[int, int] | None = None,
    video_names: list[str] | None = None,
    frame_count: bool = False,
):
    if len(sources) > tile_shape[0] * tile_shape[1]:
        raise ValueError(f"Too many sources to tile: {len(sources)} > {tile_shape[0] * tile_shape[1]}")

    frame_sources = [
        s if isinstance(s, FrameSource) else FrameSource(video_path=s)
        for s in sources
    ]

    L = frame_sources[0].n_frames
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
        if frame_count:
            frame_text = f"Frame {l}"
            (tw, th), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(img, frame_text, (W_frame - tw - 10, th + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        writer.write_frame(img)
    writer.close()
