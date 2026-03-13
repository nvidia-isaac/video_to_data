"""
MoGe image to depth processing function.
Can be called directly from command line or imported as a function.
"""
from moge.model.v2 import MoGeModel
import os
import sys
import argparse
import torch
import numpy as np
import json
from PIL import Image
from modules.common.datatypes import DepthImage, CameraIntrinsics

# Singleton model instance
_model = None

def _get_model():
    global _model
    if _model is None:
        checkpoint_dir = os.environ.get("CHECKPOINT_DIR")
        if checkpoint_dir is None:
            raise ValueError("CHECKPOINT_DIR environment variable must be set")
        
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(
                f"Checkpoint directory not found: {checkpoint_dir}\n"
                f"Please download checkpoints using: modules/moge/download.sh"
            )
        
        print(f"Initializing MoGeV2 model from {checkpoint_dir}...")
        
        # Try multiple possible checkpoint file names/locations
        possible_paths = [
            os.path.join(checkpoint_dir, "model.pt"),
            os.path.join(checkpoint_dir, "pytorch_model.bin"),
            os.path.join(checkpoint_dir, "model.safetensors"),
            checkpoint_dir,  # Some models can load from directory directly
        ]
        
        checkpoint_path = None
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            # List what files are actually in the directory
            files_in_dir = []
            if os.path.isdir(checkpoint_dir):
                files_in_dir = os.listdir(checkpoint_dir)
            
            raise FileNotFoundError(
                f"MoGeV2 checkpoint not found in {checkpoint_dir}\n"
                f"Expected one of: {[os.path.basename(p) for p in possible_paths if p != checkpoint_dir]}\n"
                f"Files found in directory: {files_in_dir}\n"
                f"Please download checkpoints using: modules/moge/download.sh"
            )
        
        print(f"Loading MoGeV2 from local checkpoint: {checkpoint_path}")
        try:
            _model = MoGeModel.from_pretrained(checkpoint_path)
            _model.to("cuda")
            _model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MoGeV2 model from {checkpoint_path}\n"
                f"Error: {e}\n"
                f"Please ensure the checkpoint is downloaded correctly using: modules/moge/download.sh"
            ) from e
    return _model

def _preprocess_numpy(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

def image_to_depth(image_path: str, depth_path: str, intrinsics_path: str):
    """Process single image to depth."""
    model = _get_model()
    image = Image.open(image_path)
    
    # Convert to RGB if needed (handles RGBA, L, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess
    img_array = np.asarray(image)
    input_tensor = _preprocess_numpy(img_array).unsqueeze(0).to("cuda")
    
    # Inference
    with torch.no_grad():
        predictions = model.infer(input_tensor)
    
    # Postprocess
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
    
    # Save
    depth_img.to_pil_image().save(depth_path)
    with open(intrinsics_path, "w") as f:
        json.dump(camera_intrinsics.to_dict(), f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image to depth")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--depth_path", type=str, required=True, help="Output path for depth image")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Output path for camera intrinsics JSON")
    
    args = parser.parse_args()
    image_to_depth(args.image_path, args.depth_path, args.intrinsics_path)

