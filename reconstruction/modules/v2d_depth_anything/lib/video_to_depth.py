"""
Depth Anything 3 video to depth processing function.
Can be called directly from command line or imported as a function.

DA3 reasons across all images passed to a single inference() call as a
multi-view scene, so video frames must be processed one at a time to obtain
independent per-frame monocular depth.
"""
import os
import argparse
import numpy as np
import cv2
from PIL import Image
from depth_anything_3.api import DepthAnything3
from v2d.common.datatypes import DepthImage, CameraIntrinsics

_model = None


def _get_model(weights_path: str) -> DepthAnything3:
    global _model
    if _model is None:
        print(f"Initializing Depth Anything 3 model from {weights_path}...")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Depth Anything 3 checkpoint not found at {weights_path}")
        _model = DepthAnything3.from_pretrained(weights_path)
        _model = _model.to("cuda")
        _model.eval()
    return _model


def video_to_depth(
    video_path: str,
    depth_folder: str,
    intrinsics_folder: str,
    weights_path: str,
    input_intrinsics_path: str = None,
):
    """Process video to depth frames using Depth Anything 3.

    Args:
        video_path:              Path to input video.
        depth_folder:            Output folder for depth images.
        intrinsics_folder:       Output folder for camera intrinsics JSON files.
        weights_path:            Path to Depth Anything 3 model weights.
        input_intrinsics_path:   Optional path to a CameraIntrinsics JSON with
                                 known calibrated intrinsics. When provided, the
                                 intrinsics are passed to DA3 as conditioning and
                                 written to the output folder instead of DA3's
                                 estimates.
    """
    model = _get_model(weights_path)
    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(intrinsics_folder, exist_ok=True)

    known_intrinsics: CameraIntrinsics | None = None
    if input_intrinsics_path is not None:
        known_intrinsics = CameraIntrinsics.load(input_intrinsics_path)

    cap = cv2.VideoCapture(video_path)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w = image.height, image.width

        infer_kwargs = {}
        if known_intrinsics is not None:
            K = np.array([[known_intrinsics.fx, 0, known_intrinsics.cx],
                          [0, known_intrinsics.fy, known_intrinsics.cy],
                          [0, 0, 1]], dtype=np.float32)
            infer_kwargs["intrinsics"] = K[None]  # (1, 3, 3)

        prediction = model.inference([image], **infer_kwargs)

        depth = np.array(prediction.depth[0])  # (H', W')
        depth_h, depth_w = depth.shape
        if depth.shape != (h, w):
            import torch
            import torch.nn.functional as F
            depth = F.interpolate(
                torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            ).squeeze().numpy()

        if known_intrinsics is not None:
            camera_intrinsics = known_intrinsics
        else:
            K = np.array(prediction.intrinsics[0])  # (3, 3) — in model's output resolution
            scale_x = w / depth_w
            scale_y = h / depth_h
            camera_intrinsics = CameraIntrinsics(
                fx=float(K[0, 0]) * scale_x,
                fy=float(K[1, 1]) * scale_y,
                cx=float(K[0, 2]) * scale_x,
                cy=float(K[1, 2]) * scale_y,
                width=w,
                height=h,
            )

        DepthImage(depth=depth).to_pil_image().save(
            os.path.join(depth_folder, f"{frame_index:06d}.png")
        )
        camera_intrinsics.save(
            os.path.join(intrinsics_folder, f"{frame_index:06d}.json")
        )

        frame_index += 1

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video to depth frames with Depth Anything 3")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--depth_folder", type=str, required=True, help="Output folder for depth images")
    parser.add_argument("--intrinsics_folder", type=str, required=True, help="Output folder for camera intrinsics")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to weights")
    parser.add_argument("--input_intrinsics_path", type=str, default=None, help="Optional known camera intrinsics JSON")

    args = parser.parse_args()
    video_to_depth(
        args.video_path, args.depth_folder, args.intrinsics_folder, args.weights_path,
        input_intrinsics_path=args.input_intrinsics_path,
    )
