import argparse
from pathlib import Path

from .video import FrameSource, tile_videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile multiple video/image sources into a single video")
    parser.add_argument("sources", nargs="+", type=str,
                        help="Paths to video files or image directories")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output video path")
    parser.add_argument("--tile_shape", type=int, nargs=2, required=True, metavar=("ROWS", "COLS"),
                        help="Tile grid shape (rows cols)")
    parser.add_argument("--names", type=str, nargs="+", default=None,
                        help="Optional label for each source")
    args = parser.parse_args()

    frame_sources = []
    for s in args.sources:
        p = Path(s)
        if p.is_dir():
            frame_sources.append(FrameSource(image_dir=p))
        else:
            frame_sources.append(FrameSource(video_path=p))

    tile_videos(
        sources=frame_sources,
        output_path=Path(args.output_path),
        tile_shape=tuple(args.tile_shape),
        video_names=args.names,
    )
