"""
Depth Anything 3 single image to depth processing function.
Can be called directly from command line or imported as a function.
"""
import os
import argparse
import numpy as np
from PIL import Image
from v2d.common.datatypes import DepthImage, CameraIntrinsics
from v2d.depth_anything.lib.video_to_depth import _get_model


def image_to_depth(
    image_path: str,
    depth_path: str,
    intrinsics_path: str,
    weights_path: str,
    input_intrinsics_path: str = None,
):
    """Process a single image to depth using Depth Anything 3.

    Args:
        image_path:              Path to input image.
        depth_path:              Output path for depth image PNG.
        intrinsics_path:         Output path for camera intrinsics JSON.
        weights_path:            Path to Depth Anything 3 model weights.
        input_intrinsics_path:   Optional path to a CameraIntrinsics JSON with
                                 known calibrated intrinsics. When provided, the
                                 intrinsics are passed to DA3 as conditioning and
                                 written as output instead of DA3's estimates.
    """
    model = _get_model(weights_path)
    os.makedirs(os.path.dirname(os.path.abspath(depth_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(intrinsics_path)), exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    h, w = image.height, image.width

    infer_kwargs = {}
    known_intrinsics: CameraIntrinsics | None = None
    if input_intrinsics_path is not None:
        known_intrinsics = CameraIntrinsics.load(input_intrinsics_path)
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

    DepthImage(depth=depth).to_pil_image().save(depth_path)
    camera_intrinsics.save(intrinsics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image to depth with Depth Anything 3")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--depth_path", type=str, required=True, help="Output path for depth image")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Output path for camera intrinsics")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to weights")
    parser.add_argument("--input_intrinsics_path", type=str, default=None, help="Optional known camera intrinsics JSON")

    args = parser.parse_args()
    image_to_depth(
        args.image_path, args.depth_path, args.intrinsics_path, args.weights_path,
        input_intrinsics_path=args.input_intrinsics_path,
    )
