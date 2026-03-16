from v2d.docker.container import run_in_container
from v2d.moge.docker._config import IMAGE_NAME, MODULES_DIR

def run_video_to_depth(video_path: str, depth_folder: str, intrinsics_folder: str, weights_path: str, batch_size: int = 8, dev: bool = False) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.moge.lib.video_to_depth",
        inputs={"video_path": video_path, "weights_path": weights_path},
        outputs={"depth_folder": depth_folder, "intrinsics_folder": intrinsics_folder},
        extra_args={"batch_size": batch_size},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video to depth")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--depth_folder", type=str, required=True, help="Output folder for depth images")
    parser.add_argument("--intrinsics_folder", type=str, required=True, help="Output folder for camera intrinsics")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to weights")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_video_to_depth(args.video_path, args.depth_folder, args.intrinsics_folder, args.weights_path, args.batch_size, dev=args.dev)
