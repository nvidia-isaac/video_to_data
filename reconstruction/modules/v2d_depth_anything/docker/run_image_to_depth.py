# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
from v2d.docker.container import run_in_container
from v2d.depth_anything.docker._config import IMAGE_NAME, MODULES_DIR


def run_image_to_depth(
    image_path: str,
    depth_path: str,
    intrinsics_path: str,
    weights_path: str,
    model: str = "nested",
    input_intrinsics_path: str = None,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
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
        extra_args={
            "model": model,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "use_ray_pose": use_ray_pose,
            "ref_view_strategy": ref_view_strategy,
        },
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
    parser.add_argument("--model", type=str, default="nested", choices=["nested", "metric"],
                        help="Model variant: 'nested' (default) or 'metric'")
    parser.add_argument("--input_intrinsics_path", type=str, default=None)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--process_res_method", type=str, default="upper_bound_resize")
    parser.add_argument("--use_ray_pose", action="store_true")
    parser.add_argument("--ref_view_strategy", type=str, default="saddle_balanced")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_image_to_depth(
        args.image_path, args.depth_path, args.intrinsics_path, args.weights_path,
        model=args.model,
        input_intrinsics_path=args.input_intrinsics_path,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
        dev=args.dev,
    )
