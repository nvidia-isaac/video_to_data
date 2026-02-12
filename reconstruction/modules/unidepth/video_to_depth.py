"""
UniDepth video to depth processing function.
Can be called directly from command line or imported as a function.
"""
from unidepth.models import UniDepthV2
import os
import sys
import argparse
import torch
import numpy as np
import cv2
import json
from PIL import Image
from modules.common.datatypes import DepthImage, CameraIntrinsics

# Singleton model instance
_model = None

def _get_model():
    global _model
    if _model is None:
        data_dir = os.environ.get("DATA_DIR", "/data")
        checkpoint_dir = os.environ.get("CHECKPOINT_DIR", os.path.join(data_dir, "unidepth/checkpoints/unidepth-v2-vitl14"))
        
        print(f"Initializing UniDepthV2 model from {checkpoint_dir}...")
        if os.path.exists(checkpoint_dir):
            print(f"Loading UniDepthV2 from local checkpoint: {checkpoint_dir}")
            _model = UniDepthV2.from_pretrained(checkpoint_dir)
        else:
            raise FileNotFoundError(f"UniDepthV2 checkpoint not found at {checkpoint_dir}")
        _model.to("cuda")
        _model.eval()
    return _model

def _preprocess_numpy(image: np.ndarray) -> torch.Tensor:
    """Preprocess numpy image to tensor"""
    return torch.from_numpy(image).permute(2, 0, 1).float()

def video_to_depth(video_path: str, depth_folder: str, intrinsics_folder: str, batch_size: int = 8):
    """Process video to depth frames."""
    model = _get_model()
    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(intrinsics_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    image_batch = []
    
    for frame_index in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image_batch.append(image)
        
        if len(image_batch) == batch_size or frame_index == total_frames - 1:
            # Process batch
            input_tensor = torch.stack([
                _preprocess_numpy(np.asarray(img)) for img in image_batch
            ], dim=0).to("cuda")
            
            with torch.no_grad():
                predictions = model.infer(input_tensor)
            
            for i, img in enumerate(image_batch):
                depth = predictions["depth"][i,0].cpu().numpy()
                intrinsics = predictions["intrinsics"][i].cpu().numpy()
                
                depth_img = DepthImage(depth=depth)
                camera_intrinsics = CameraIntrinsics(
                    fx=float(intrinsics[0, 0]),
                    fy=float(intrinsics[1, 1]),
                    cx=float(intrinsics[0, 2]),
                    cy=float(intrinsics[1, 2]),
                    width=img.width,
                    height=img.height
                )
                
                depth_img.to_pil_image().save(
                    os.path.join(depth_folder, f"{frame_index - len(image_batch) + i + 1:06d}.png")
                )
                with open(
                    os.path.join(intrinsics_folder, f"{frame_index - len(image_batch) + i + 1:06d}.json"), "w"
                ) as f:
                    json.dump(camera_intrinsics.to_dict(), f, indent=4)
            
            image_batch = []
    
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video to depth frames")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--depth_folder", type=str, required=True, help="Output folder for depth images")
    parser.add_argument("--intrinsics_folder", type=str, required=True, help="Output folder for camera intrinsics")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    
    args = parser.parse_args()
    video_to_depth(args.video_path, args.depth_folder, args.intrinsics_folder, args.batch_size)

