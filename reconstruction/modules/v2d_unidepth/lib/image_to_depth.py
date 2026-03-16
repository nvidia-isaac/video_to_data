"""
UniDepth image to depth processing function.
Can be called directly from command line or imported as a function.
"""
from unidepth.models import UniDepthV2
import os
import argparse
import torch
import numpy as np
from PIL import Image
from v2d.common.datatypes import DepthImage, CameraIntrinsics

# Singleton model instance
_model = None

def _get_model(weights_path: str):
    global _model
    if _model is None:
        print(f"Initializing UniDepthV2 model from {weights_path}...")
        if os.path.exists(weights_path):
            print(f"Loading UniDepthV2 from local checkpoint: {weights_path}")
            _model = UniDepthV2.from_pretrained(weights_path)
        else:
            raise FileNotFoundError(f"UniDepthV2 checkpoint not found at {weights_path}")
        _model.to("cuda")
        _model.eval()
    return _model

def _preprocess_numpy(image: np.ndarray) -> torch.Tensor:
    """Preprocess numpy image to tensor"""
    return torch.from_numpy(image).permute(2, 0, 1).float()

def image_to_depth(image_path: str, depth_path: str, intrinsics_path: str, weights_path: str):
    """Process a single image to depth."""
    model = _get_model(weights_path)
    
    img = Image.open(image_path).convert("RGB")
    input_tensor = _preprocess_numpy(np.asarray(img)).unsqueeze(0).to("cuda")
    
    with torch.no_grad():
        predictions = model.infer(input_tensor)
    
    depth = predictions["depth"][0, 0].cpu().numpy()
    intrinsics = predictions["intrinsics"][0].cpu().numpy()
    
    depth_img = DepthImage(depth=depth)
    camera_intrinsics = CameraIntrinsics(
        fx=float(intrinsics[0, 0]),
        fy=float(intrinsics[1, 1]),
        cx=float(intrinsics[0, 2]),
        cy=float(intrinsics[1, 2]),
        width=img.width,
        height=img.height
    )
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(depth_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(intrinsics_path)), exist_ok=True)
    
    depth_img.to_pil_image().save(depth_path)
    camera_intrinsics.save(intrinsics_path)
    
    print(f"Saved depth to {depth_path} and intrinsics to {intrinsics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process single image to depth")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--depth_path", type=str, required=True, help="Output path for depth image")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Output path for camera intrinsics")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to weights")
    args = parser.parse_args()
    image_to_depth(args.image_path, args.depth_path, args.intrinsics_path, args.weights_path)

