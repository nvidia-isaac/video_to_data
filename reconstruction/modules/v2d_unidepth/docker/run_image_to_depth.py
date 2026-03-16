from v2d.docker.container import run_in_container
from v2d.unidepth.docker._config import IMAGE_NAME, MODULES_DIR

def run_image_to_depth(image_path: str, depth_path: str, intrinsics_path: str, weights_path: str, dev: bool = False) -> None:
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.unidepth.lib.image_to_depth",
        inputs={"image_path": image_path, "weights_path": weights_path},
        outputs={"depth_path": depth_path, "intrinsics_path": intrinsics_path},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process image to depth")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--depth_path", type=str, required=True, help="Output path for depth image")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Output path for camera intrinsics")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to weights")
    parser.add_argument("--dev", action="store_true", help="Mount local modules for development")
    args = parser.parse_args()
    run_image_to_depth(args.image_path, args.depth_path, args.intrinsics_path, args.weights_path, dev=args.dev)
