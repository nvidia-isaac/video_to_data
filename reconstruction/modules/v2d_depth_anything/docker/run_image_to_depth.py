from v2d.docker.container import run_in_container
from v2d.depth_anything.docker._config import IMAGE_NAME, MODULES_DIR


def run_image_to_depth(
    image_path: str,
    depth_path: str,
    intrinsics_path: str,
    weights_path: str,
    input_intrinsics_path: str = None,
    dev: bool = False,
) -> None:
    inputs = {"image_path": image_path, "weights_path": weights_path}
    if input_intrinsics_path is not None:
        inputs["input_intrinsics_path"] = input_intrinsics_path
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.depth_anything.lib.image_to_depth",
        inputs=inputs,
        outputs={"depth_path": depth_path, "intrinsics_path": intrinsics_path},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process image to depth with Depth Anything 3")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=True)
    parser.add_argument("--intrinsics_path", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--input_intrinsics_path", type=str, default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_image_to_depth(
        args.image_path, args.depth_path, args.intrinsics_path, args.weights_path,
        input_intrinsics_path=args.input_intrinsics_path,
        dev=args.dev,
    )
