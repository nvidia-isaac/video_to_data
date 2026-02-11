import os
import cv2
import torch
import torchvision # Ensure torchvision is imported before torch.jit.load
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple
from modules.nlf.datatypes import CameraIntrinsics

# Singleton model instance
_model = None

def _get_model():
    global _model
    if _model is None:
        weights_path = os.environ.get("NLF_WEIGHTS_PATH", "modules/nlf/data/weights/nlf_l_multi_0.3.2.torchscript")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"NLF weights not found at {weights_path}. Please run download.sh or place weights manually.")
        _model = torch.jit.load(weights_path).cuda().eval()
    return _model

def masks2bbox(masks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box from a list of masks."""
    all_idx = []
    for mask in masks:
        idx = np.where(mask > 127)
        if len(idx[0]) > 0:
            all_idx.append(np.stack(idx, axis=1))
    
    if len(all_idx) == 0:
        return np.array([0, 0]), np.array([0, 0])
    
    all_idx = np.concatenate(all_idx, axis=0)
    bmin = np.min(all_idx, axis=0)[::-1]  # (x, y)
    bmax = np.max(all_idx, axis=0)[::-1]  # (x, y)
    return bmin, bmax

import h5py

def video_to_vertices(video_path: str, masks_dir: str, intrinsics_path: str, output_path: str = None, chunk_size: int = 32) -> np.ndarray:
    """
    Runs NLF to get 3D body vertices in camera space.
    Returns: (T, 6890, 3) vertices in meters.
    """
    # Load intrinsics from file path
    import json
    with open(intrinsics_path, 'r') as f:
        intrinsics_dict = json.load(f)
    intrinsics = CameraIntrinsics(**intrinsics_dict)

    model = _get_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_vertices = []
    
    K = intrinsics.to_matrix()
    K_tensor = torch.from_numpy(K).cuda().float()[None]

    for i in tqdm(range(0, frame_count, chunk_size), desc="Running NLF Localization"):
        images_chunk = []
        bboxes_chunk = []
        valid_indices = []

        for j in range(i, min(i + chunk_size, frame_count)):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx = j
            mask_path = os.path.join(masks_dir, f"{frame_idx:06d}.png")
            
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for frame {frame_idx}, skipping.")
                continue

            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            
            # Ensure mask is 0-255
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)

            bmin, bmax = masks2bbox([mask])
            if np.all(bmin == 0) and np.all(bmax == 0):
                print(f"Warning: No person found in mask for frame {frame_idx}, skipping.")
                continue

            bbox = np.concatenate([bmin, bmax - bmin])  # xywh
            bbox = np.append(bbox, [1.01])  # confidence
            
            images_chunk.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            bboxes_chunk.append(torch.from_numpy(bbox)[None].cuda().float())
            valid_indices.append(j)

        if not images_chunk:
            continue

        image_tensor = torch.from_numpy(np.stack(images_chunk)).permute(0, 3, 1, 2).cuda()
        
        with torch.no_grad():
            pred = model.detect_smpl_batched(
                image_tensor, 
                extra_boxes=bboxes_chunk, 
                detector_threshold=1.0, 
                suppress_implausible_poses=False, 
                intrinsic_matrix=K_tensor
            )

        # Vertices are in mm, convert to meters
        verts = torch.cat(pred['vertices3d'], dim=0).cpu().numpy() / 1000.0
        all_vertices.append(verts)

    cap.release()
    
    if not all_vertices:
        res = np.array([])
    else:
        res = np.concatenate(all_vertices, axis=0)
    
    if output_path and res.size > 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('vertices', data=res)
        print(f"Vertices saved to {output_path} (HDF5)")
        
    return res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NLF localization to get 3D vertices.")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--intrinsics_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=32)
    
    args = parser.parse_args()
    
    video_to_vertices(args.video_path, args.masks_dir, args.intrinsics_path, args.output_path, args.chunk_size)

