import argparse
from pathlib import Path

from v2d.mv.io.video import FrameSource, tile_videos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tile multiple video/image sources into a single video")
    # parser.add_argument("sources", nargs="+", type=str,
    #                     help="Paths to video files or image directories")
    # parser.add_argument("--output_path", type=str, required=True,
    #                     help="Output video path")
    # parser.add_argument("--tile_shape", type=int, nargs=2, required=True, metavar=("ROWS", "COLS"),
    #                     help="Tile grid shape (rows cols)")
    # parser.add_argument("--names", type=str, nargs="+", default=None,
    #                     help="Optional label for each source")
    # args = parser.parse_args()
    
    root_dir = "/home/dzou/datasets/v2d/proc_2026-03-12_16-58-57_tall_bar_stool_criss_cross_apple_sauce_01/sam3d_body"
    
    sources = [
        f"{root_dir}/mhr_mesh_opt_0.mp4",
        f"{root_dir}/mhr_mesh_opt_1.mp4",
        f"{root_dir}/mhr_mesh_opt_2.mp4",
        f"{root_dir}/mhr_mesh_opt_3.mp4",
        f"{root_dir}/chamfer_vis/front_stereo_camera_left.mp4",
        f"{root_dir}/chamfer_vis/back_stereo_camera_left.mp4",
        f"{root_dir}/chamfer_vis/left_stereo_camera_left.mp4",
        f"{root_dir}/chamfer_vis/right_stereo_camera_left.mp4",
    ]
    names = [
        "Front", "Back", "Left", "Right",
        "Front", "Back", "Left", "Right",
    ]
    tile_shape = (2, 4)
    output_path = f"{root_dir}/tiled_videos.mp4"

    frame_sources = []
    for s in sources:
        p = Path(s)
        if p.is_dir():
            frame_sources.append(FrameSource(image_dir=p))
        else:
            frame_sources.append(FrameSource(video_path=p))

    tile_videos(
        sources=frame_sources,
        output_path=Path(output_path),
        tile_shape=tuple(tile_shape),
        output_image_size=(768, 576),
        video_names=names,
    )
