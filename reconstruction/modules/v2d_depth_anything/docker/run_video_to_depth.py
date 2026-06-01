# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.depth_anything.docker._config import IMAGE_NAME, MODULES_DIR


def run_video_to_depth(
    video_path: str,
    depth_folder: str,
    intrinsics_folder: str,
    weights_path: str,
    model: str = "nested",
    input_intrinsics_path: str = None,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
    chunk_size: int = 0,
    chunk_overlap: int = 10,
    dev: bool = False,
) -> None:
    inputs = {"video_path": video_path, "weights_path": weights_path}
    if input_intrinsics_path is not None:
        inputs["input_intrinsics_path"] = input_intrinsics_path
    run_in_container(
        image=IMAGE_NAME,
        module="v2d.depth_anything.lib.video_to_depth",
        inputs=inputs,
        outputs={"depth_folder": depth_folder, "intrinsics_folder": intrinsics_folder},
        extra_args={
            "model": model,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "use_ray_pose": use_ray_pose,
            "ref_view_strategy": ref_view_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process video to depth with Depth Anything 3")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--depth_folder", type=str, required=True)
    parser.add_argument("--intrinsics_folder", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="nested", choices=["nested", "metric"],
                        help="Model variant: 'nested' (default) or 'metric'")
    parser.add_argument("--input_intrinsics_path", type=str, default=None)
    parser.add_argument("--process_res", type=int, default=504)
    parser.add_argument("--process_res_method", type=str, default="upper_bound_resize")
    parser.add_argument("--use_ray_pose", action="store_true")
    parser.add_argument("--ref_view_strategy", type=str, default="saddle_balanced")
    parser.add_argument("--chunk_size", type=int, default=80)
    parser.add_argument("--chunk_overlap", type=int, default=10)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    run_video_to_depth(
        args.video_path, args.depth_folder, args.intrinsics_folder, args.weights_path,
        model=args.model,
        input_intrinsics_path=args.input_intrinsics_path,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        dev=args.dev,
    )
