# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from v2d.docker.container import run_in_container
from v2d.droid_slam.docker._config import IMAGE_NAME, MODULES_DIR


def run_image_list_to_slam(
    image_folder: str,
    poses_folder: str,
    weights_path: str,
    input_intrinsics_path: str = None,
    input_intrinsics_folder: str = None,
    input_depth_folder: str = None,
    align_to_depth_folder: str = None,
    depth_folder: str = None,
    pointcloud_path: str = None,
    trajectory_path: str = None,
    stride: int = 1,
    image_height: int = 384,
    image_width: int = 512,
    buffer_size: int = 512,
    beta: float = 0.3,
    filter_thresh: float = 2.4,
    warmup: int = 8,
    keyframe_thresh: float = 4.0,
    frontend_thresh: float = 16.0,
    frontend_window: int = 25,
    frontend_radius: int = 2,
    frontend_nms: int = 1,
    backend_thresh: float = 22.0,
    backend_radius: int = 2,
    backend_nms: int = 3,
    upsample: bool = False,
    pointcloud_min_views: int = 2,
    dev: bool = False,
) -> None:
    inputs = {"image_folder": image_folder, "weights_path": weights_path}
    if input_intrinsics_path is not None:
        inputs["input_intrinsics_path"] = input_intrinsics_path
    if input_intrinsics_folder is not None:
        inputs["input_intrinsics_folder"] = input_intrinsics_folder
    if input_depth_folder is not None:
        inputs["input_depth_folder"] = input_depth_folder
    if align_to_depth_folder is not None:
        inputs["align_to_depth_folder"] = align_to_depth_folder

    outputs = {"poses_folder": poses_folder}
    if depth_folder is not None:
        outputs["depth_folder"] = depth_folder
    if pointcloud_path is not None:
        outputs["pointcloud_path"] = pointcloud_path
    if trajectory_path is not None:
        outputs["trajectory_path"] = trajectory_path

    extra_args = {
        "stride": stride,
        "image_height": image_height,
        "image_width": image_width,
        "buffer_size": buffer_size,
        "beta": beta,
        "filter_thresh": filter_thresh,
        "warmup": warmup,
        "keyframe_thresh": keyframe_thresh,
        "frontend_thresh": frontend_thresh,
        "frontend_window": frontend_window,
        "frontend_radius": frontend_radius,
        "frontend_nms": frontend_nms,
        "backend_thresh": backend_thresh,
        "backend_radius": backend_radius,
        "backend_nms": backend_nms,
        "upsample": upsample,
        "pointcloud_min_views": pointcloud_min_views,
    }

    run_in_container(
        image=IMAGE_NAME,
        module="v2d.droid_slam.lib.image_list_to_slam",
        inputs=inputs,
        outputs=outputs,
        extra_args=extra_args,
        dev=dev,
        modules_dir=MODULES_DIR,
        gpus=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DROID-SLAM on a folder of frames")
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--poses_folder", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--input_intrinsics_path", default=None)
    parser.add_argument("--input_intrinsics_folder", default=None)
    parser.add_argument("--input_depth_folder", default=None,
                        help="Per-frame metric depth fed into BA as a sensor prior.")
    parser.add_argument("--align_to_depth_folder", default=None,
                        help="Per-frame metric depth used post-hoc for scale alignment only.")
    parser.add_argument("--depth_folder", default=None)
    parser.add_argument("--pointcloud_path", default=None)
    parser.add_argument("--trajectory_path", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image_height", type=int, default=384)
    parser.add_argument("--image_width", type=int, default=512)
    parser.add_argument("--buffer_size", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--filter_thresh", type=float, default=2.4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--keyframe_thresh", type=float, default=4.0)
    parser.add_argument("--frontend_thresh", type=float, default=16.0)
    parser.add_argument("--frontend_window", type=int, default=25)
    parser.add_argument("--frontend_radius", type=int, default=2)
    parser.add_argument("--frontend_nms", type=int, default=1)
    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--pointcloud_min_views", type=int, default=2)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    run_image_list_to_slam(
        image_folder=args.image_folder,
        poses_folder=args.poses_folder,
        weights_path=args.weights_path,
        input_intrinsics_path=args.input_intrinsics_path,
        input_intrinsics_folder=args.input_intrinsics_folder,
        input_depth_folder=args.input_depth_folder,
        align_to_depth_folder=args.align_to_depth_folder,
        depth_folder=args.depth_folder,
        pointcloud_path=args.pointcloud_path,
        trajectory_path=args.trajectory_path,
        stride=args.stride,
        image_height=args.image_height,
        image_width=args.image_width,
        buffer_size=args.buffer_size,
        beta=args.beta,
        filter_thresh=args.filter_thresh,
        warmup=args.warmup,
        keyframe_thresh=args.keyframe_thresh,
        frontend_thresh=args.frontend_thresh,
        frontend_window=args.frontend_window,
        frontend_radius=args.frontend_radius,
        frontend_nms=args.frontend_nms,
        backend_thresh=args.backend_thresh,
        backend_radius=args.backend_radius,
        backend_nms=args.backend_nms,
        upsample=args.upsample,
        pointcloud_min_views=args.pointcloud_min_views,
        dev=args.dev,
    )
