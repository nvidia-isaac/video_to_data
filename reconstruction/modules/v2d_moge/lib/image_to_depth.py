# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MoGe image to depth processing function.
Can be called directly from command line or imported as a function.
"""
from moge.model.v2 import MoGeModel
import os
import argparse
import torch
import numpy as np
from PIL import Image
from v2d.common.datatypes import DepthImage, CameraIntrinsics

_model = None

def _get_model(weights_path: str):
    global _model
    if _model is None:
        print(f"Initializing MoGeV2 model from {weights_path}...")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"MoGeV2 checkpoint not found at {weights_path}")

        possible_paths = [
            os.path.join(weights_path, "model.pt"),
            os.path.join(weights_path, "pytorch_model.bin"),
            os.path.join(weights_path, "model.safetensors"),
            weights_path,
        ]

        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break

        if checkpoint_path is None:
            files_in_dir = os.listdir(weights_path) if os.path.isdir(weights_path) else []
            raise FileNotFoundError(
                f"MoGeV2 checkpoint not found in {weights_path}\n"
                f"Files found: {files_in_dir}"
            )

        print(f"Loading MoGeV2 from local checkpoint: {checkpoint_path}")
        _model = MoGeModel.from_pretrained(checkpoint_path)
        _model.to("cuda")
        _model.eval()
    return _model

def _preprocess_numpy(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

def image_to_depth(
    image_path: str,
    depth_path: str,
    intrinsics_path: str,
    weights_path: str,
    input_intrinsics_path: str = None,
    points_path: str = None,
    normals_path: str = None,
    mask_path: str = None,
):
    """Process single image to depth.

    Args:
        image_path:            Input image path.
        depth_path:            Output path for depth PNG.
        intrinsics_path:       Output path for intrinsics JSON.
        weights_path:          MoGe weights directory.
        input_intrinsics_path: Optional CameraIntrinsics JSON. When given, the
                               horizontal FoV is derived from fx and passed to
                               MoGe as a focal-length prior; the intrinsics
                               written to disk are still MoGe's returned
                               values (geometrically consistent with depth).
    """
    model = _get_model(weights_path)
    image = Image.open(image_path)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.asarray(image)
    input_tensor = _preprocess_numpy(img_array).unsqueeze(0).to("cuda")

    fov_x: float | None = None
    if input_intrinsics_path is not None:
        known = CameraIntrinsics.load(input_intrinsics_path)
        fov_x = float(np.degrees(2 * np.arctan(known.width / (2 * known.fx))))
        print(f"  Using fov_x={fov_x:.2f}° from {input_intrinsics_path}")

    with torch.no_grad():
        predictions = model.infer(input_tensor, fov_x=fov_x)

    depth = predictions["depth"][0].cpu().numpy()
    intrinsics = predictions["intrinsics"][0].cpu().numpy()

    depth_img = DepthImage(depth=depth)
    camera_intrinsics = CameraIntrinsics(
        fx=float(intrinsics[0, 0] * image.width),
        fy=float(intrinsics[1, 1] * image.height),
        cx=float(intrinsics[0, 2] * image.width),
        cy=float(intrinsics[1, 2] * image.height),
        width=image.width,
        height=image.height
    )

    os.makedirs(os.path.dirname(os.path.abspath(depth_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(intrinsics_path)), exist_ok=True)

    depth_img.to_pil_image().save(depth_path)
    camera_intrinsics.save(intrinsics_path)

    if points_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(points_path)), exist_ok=True)
        np.save(points_path, predictions["points"][0].cpu().numpy())
    if normals_path is not None and "normal" in predictions:
        os.makedirs(os.path.dirname(os.path.abspath(normals_path)), exist_ok=True)
        np.save(normals_path, predictions["normal"][0].cpu().numpy())
    if mask_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(mask_path)), exist_ok=True)
        mask_arr = (predictions["mask"][0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(mask_arr).save(mask_path)

    print(f"Saved depth to {depth_path} and intrinsics to {intrinsics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image to depth")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--depth_path", type=str, required=True, help="Output path for depth image")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Output path for camera intrinsics JSON")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to weights")
    parser.add_argument("--input_intrinsics_path", type=str, default=None, help="Optional known camera intrinsics JSON (used as fov_x prior)")

    parser.add_argument("--points_path", type=str, default=None, help="Output path for pointmap .npy (H,W,3 float32)")
    parser.add_argument("--normals_path", type=str, default=None, help="Output path for surface normals .npy (H,W,3 float32)")
    parser.add_argument("--mask_path", type=str, default=None, help="Output path for validity mask PNG")

    args = parser.parse_args()
    image_to_depth(
        args.image_path, args.depth_path, args.intrinsics_path, args.weights_path,
        input_intrinsics_path=args.input_intrinsics_path,
        points_path=args.points_path,
        normals_path=args.normals_path,
        mask_path=args.mask_path,
    )
