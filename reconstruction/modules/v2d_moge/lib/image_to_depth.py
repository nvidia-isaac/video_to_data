"""
MoGe image to depth processing function.
Can be called directly from command line or imported as a function.
"""
from moge.model.v2 import MoGeModel
import os
import argparse
import torch
import numpy as np
import json
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

def image_to_depth(image_path: str, depth_path: str, intrinsics_path: str, weights_path: str):
    """Process single image to depth."""
    model = _get_model(weights_path)
    image = Image.open(image_path)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.asarray(image)
    input_tensor = _preprocess_numpy(img_array).unsqueeze(0).to("cuda")

    with torch.no_grad():
        predictions = model.infer(input_tensor)

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
    with open(intrinsics_path, "w") as f:
        json.dump(camera_intrinsics.to_dict(), f, indent=4)

    print(f"Saved depth to {depth_path} and intrinsics to {intrinsics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image to depth")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--depth_path", type=str, required=True, help="Output path for depth image")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Output path for camera intrinsics JSON")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to weights")

    args = parser.parse_args()
    image_to_depth(args.image_path, args.depth_path, args.intrinsics_path, args.weights_path)
