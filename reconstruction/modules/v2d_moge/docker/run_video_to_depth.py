from v2d.docker.container import run_in_container
from v2d.moge.docker._config import IMAGE_NAME, MODULES_DIR


def run_video_to_depth(
    video_path: str,
    depth_folder: str,
    intrinsics_folder: str,
    weights_path: str,
    batch_size: int = 8,
    input_intrinsics_path: str = None,
    dev: bool = False,
) -> None:
    inputs = {"video_path": video_path, "weights_path": weights_path}
    if input_intrinsics_path is not None:
        inputs["input_intrinsics_path"] = input_intrinsics_path
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.moge.lib.video_to_depth",
        inputs=inputs,
        outputs={"depth_folder": depth_folder, "intrinsics_folder": intrinsics_folder},
        extra_args={"batch_size": batch_size},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video to depth")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--depth_folder", type=str, required=True)
    parser.add_argument("--intrinsics_folder", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_intrinsics_path", type=str, default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_video_to_depth(
        args.video_path, args.depth_folder, args.intrinsics_folder, args.weights_path,
        batch_size=args.batch_size,
        input_intrinsics_path=args.input_intrinsics_path,
        dev=args.dev,
    )
