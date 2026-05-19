"""
Run DROID-SLAM on a video file → per-frame camera poses (+ optional depth, PLY).

Intrinsics can be supplied either as a single CameraIntrinsics JSON
(--input_intrinsics_path; reused for every frame) or as a folder of
per-frame intrinsics JSON files (--input_intrinsics_folder; values are
median-pooled into a single set, since DROID-SLAM uses a fixed intrinsic
for the whole sequence).

If --input_depth_folder is provided, per-frame depth (e.g. from MoGe) is
fed into DROID-SLAM's bundle adjustment as a metric prior — this directly
shapes the reconstructed geometry rather than rescaling it after the fact.

If --align_to_depth_folder is provided, the reconstructed poses and depths
are scale-aligned to that metric reference via a robust per-keyframe ratio
fit. Pure post-processing — does not affect SLAM behaviour.

The two flags are independent and may be combined: --input_depth_folder
shapes BA, --align_to_depth_folder rescales output. With a metric prior
provided, post-hoc rescaling is usually unnecessary.
"""
from __future__ import annotations

import argparse
import os
from typing import Iterator

import cv2
import numpy as np

from v2d.common.datatypes import CameraIntrinsics, DepthImage
from v2d.droid_slam.lib._slam import Frame, run_droid_slam


def _load_intrinsics(
    input_intrinsics_path: str | None,
    input_intrinsics_folder: str | None,
) -> CameraIntrinsics:
    if input_intrinsics_path is None and input_intrinsics_folder is None:
        raise ValueError("Provide either --input_intrinsics_path or --input_intrinsics_folder")
    if input_intrinsics_path is not None:
        return CameraIntrinsics.load(input_intrinsics_path)
    # Per-frame intrinsics → median pool into one shared intrinsic for DROID.
    files = sorted(
        os.path.join(input_intrinsics_folder, f)
        for f in os.listdir(input_intrinsics_folder)
        if f.endswith(".json")
    )
    if not files:
        raise FileNotFoundError(f"No intrinsics JSONs in {input_intrinsics_folder}")
    fx, fy, cx, cy, w, h = [], [], [], [], [], []
    for fp in files:
        ci = CameraIntrinsics.load(fp)
        fx.append(ci.fx); fy.append(ci.fy); cx.append(ci.cx); cy.append(ci.cy)
        w.append(ci.width); h.append(ci.height)
    return CameraIntrinsics(
        fx=float(np.median(fx)), fy=float(np.median(fy)),
        cx=float(np.median(cx)), cy=float(np.median(cy)),
        width=int(np.median(w)), height=int(np.median(h)),
    )


def _depth_for_frame(folder: str | None, frame_idx: int) -> np.ndarray | None:
    if folder is None:
        return None
    path = os.path.join(folder, f"{frame_idx:06d}.png")
    if not os.path.exists(path):
        return None
    return DepthImage.load(path).depth.astype(np.float32)


def video_to_slam(
    video_path: str,
    poses_folder: str,
    weights_path: str,
    input_intrinsics_path: str | None = None,
    input_intrinsics_folder: str | None = None,
    input_depth_folder: str | None = None,
    align_to_depth_folder: str | None = None,
    depth_folder: str | None = None,
    pointcloud_path: str | None = None,
    trajectory_path: str | None = None,
    stride: int = 1,
    image_size: tuple[int, int] = (384, 512),
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
) -> None:
    intrinsics = _load_intrinsics(input_intrinsics_path, input_intrinsics_folder)

    def frame_iter() -> Iterator[Frame]:
        cap = cv2.VideoCapture(video_path)
        try:
            i = 0
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if i % stride == 0:
                    yield Frame(
                        frame_idx=i,
                        image_bgr=frame_bgr,
                        intrinsics=intrinsics,
                        prior_depth=_depth_for_frame(input_depth_folder, i),
                        align_to_depth=_depth_for_frame(align_to_depth_folder, i),
                    )
                i += 1
        finally:
            cap.release()

    run_droid_slam(
        frame_iter=frame_iter,
        weights_path=weights_path,
        poses_folder=poses_folder,
        depth_folder=depth_folder,
        pointcloud_path=pointcloud_path,
        trajectory_path=trajectory_path,
        image_size=image_size,
        buffer_size=buffer_size,
        beta=beta,
        filter_thresh=filter_thresh,
        warmup=warmup,
        keyframe_thresh=keyframe_thresh,
        frontend_thresh=frontend_thresh,
        frontend_window=frontend_window,
        frontend_radius=frontend_radius,
        frontend_nms=frontend_nms,
        backend_thresh=backend_thresh,
        backend_radius=backend_radius,
        backend_nms=backend_nms,
        upsample=upsample,
        pointcloud_min_views=pointcloud_min_views,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DROID-SLAM on a video")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--poses_folder", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--input_intrinsics_path", default=None)
    parser.add_argument("--input_intrinsics_folder", default=None)
    parser.add_argument("--input_depth_folder", default=None,
                        help="Per-frame metric depth fed into BA as a sensor prior "
                             "(e.g. MoGe output). Shapes the reconstruction.")
    parser.add_argument("--align_to_depth_folder", default=None,
                        help="Per-frame metric depth used post-hoc for scale "
                             "alignment only. Does not influence BA.")
    parser.add_argument("--depth_folder", default=None,
                        help="Optional output: per-keyframe DROID depth (scale-aligned)")
    parser.add_argument("--pointcloud_path", default=None,
                        help="Optional output: fused .ply of keyframe points")
    parser.add_argument("--trajectory_path", default=None,
                        help="Optional output: TUM-format trajectory .txt")
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
    parser.add_argument("--pointcloud_min_views", type=int, default=2,
                        help="Multi-view consistency threshold for PLY output. "
                             "1 = keep all pixels above the disparity threshold "
                             "(dense, noisier); 2 = upstream default (clean, sparse).")
    args = parser.parse_args()

    video_to_slam(
        video_path=args.video_path,
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
        image_size=(args.image_height, args.image_width),
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
    )
