"""
Align depth images to SMPL rendered depth images.
Provides absolute depth scale by computing robust scale and shift.
"""
import os
import numpy as np
import torch
from tqdm import tqdm
import glob

from v2d.foundation_pose.lib.fp_utils import erode_depth, bilateral_filter_depth

from v2d.datatypes import DepthImage, Mask


def compute_scale_and_shift_robust(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Robust scale and shift computation (https://arxiv.org/pdf/1907.01341)."""
    t_pr = np.median(pred[mask])
    t_tar = np.median(target[mask])
    scale_pr = np.mean(np.abs(pred[mask] - t_pr))
    scale_tar = np.mean(np.abs(target[mask] - t_tar))
    scale = scale_tar / scale_pr if scale_pr > 1e-6 else 1.0
    shift = t_tar - scale * t_pr
    return scale, shift


def align_depth_to_smpl(
    depth_folder: str,
    smpl_depth_folder: str,
    output_depth_folder: str,
    masks_folder: str,
    smpl_masks_folder: str = None
):
    """Align depth images to SMPL rendered depth images."""
    device = "cuda"
    os.makedirs(output_depth_folder, exist_ok=True)
    
    depth_files = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
    if len(depth_files) == 0:
        raise ValueError(f"No depth images found in {depth_folder}")
    
    print(f"Found {len(depth_files)} depth images")
    
    scales = []
    shifts = []
    
    for depth_path in tqdm(depth_files, desc="Aligning depth to SMPL"):
        depth_filename = os.path.basename(depth_path)
        smpl_depth_path = os.path.join(smpl_depth_folder, depth_filename)
        
        if not os.path.exists(smpl_depth_path):
            print(f"Warning: SMPL depth not found for {depth_filename}, skipping")
            continue
        
        depth_img = DepthImage.load(depth_path)
        depth = depth_img.depth
        
        smpl_depth_img = DepthImage.load(smpl_depth_path)
        smpl_depth = smpl_depth_img.depth
        
        depth = np.clip(depth, 0.0, 50.0)
        smpl_depth = np.clip(smpl_depth, 0.0, 50.0)
        depth[depth < 0.001] = 0.0
        smpl_depth[smpl_depth < 0.001] = 0.0
        
        depth_tensor = torch.as_tensor(depth, device=device, dtype=torch.float32)
        depth_tensor = erode_depth(depth_tensor, radius=2, device=device)
        depth_tensor = bilateral_filter_depth(depth_tensor, radius=2, device=device)
        depth_filtered = depth_tensor.cpu().numpy()
        
        smpl_depth_tensor = torch.as_tensor(smpl_depth, device=device, dtype=torch.float32)
        smpl_depth_tensor = erode_depth(smpl_depth_tensor, radius=2, device=device)
        smpl_depth_tensor = bilateral_filter_depth(smpl_depth_tensor, radius=2, device=device)
        smpl_depth_filtered = smpl_depth_tensor.cpu().numpy()
        
        mask_path = os.path.join(masks_folder, depth_filename)
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {depth_filename}, skipping")
            continue
        
        input_mask = Mask.load(mask_path).mask > 0
        
        if smpl_masks_folder is None:
            smpl_masks_folder = smpl_depth_folder
        
        smpl_mask_path = os.path.join(smpl_masks_folder, depth_filename)
        if not os.path.exists(smpl_mask_path):
            print(f"Warning: SMPL rendered mask not found for {depth_filename}, using depth-based mask")
            smpl_rendered_mask = smpl_depth_filtered > 0.001
        else:
            smpl_rendered_mask = Mask.load(smpl_mask_path).mask > 0
        
        MIN_VALID_DEPTH = 0.15
        input_valid_depth = depth_filtered > MIN_VALID_DEPTH
        smpl_valid_depth = smpl_depth_filtered > MIN_VALID_DEPTH
        
        alignment_mask = (
            input_mask & 
            smpl_rendered_mask & 
            input_valid_depth & 
            smpl_valid_depth
        )
        
        if np.sum(alignment_mask) < 100:
            print(f"Warning: Too few valid pixels ({np.sum(alignment_mask)}) for {depth_filename}, skipping")
            continue
        
        scale, shift = compute_scale_and_shift_robust(
            depth_filtered, 
            smpl_depth_filtered, 
            alignment_mask
        )
        
        if scale < 0.01 or scale > 100.0:
            valid_depth = depth_filtered[alignment_mask]
            valid_smpl_depth = smpl_depth_filtered[alignment_mask]
            print(f"[WARNING] Suspicious scale {scale:.6f} for {depth_filename}")
            print(f"  Monocular depth range: [{valid_depth.min():.3f}, {valid_depth.max():.3f}]")
            print(f"  SMPL depth range: [{valid_smpl_depth.min():.3f}, {valid_smpl_depth.max():.3f}]")
        
        scales.append(scale)
        shifts.append(shift)
        
        depth_aligned = depth.copy()
        valid_mask = depth > 0.001
        depth_aligned[valid_mask] = depth_aligned[valid_mask] * scale + shift
        depth_aligned = np.clip(depth_aligned, 0.1, 50.0)
        
        depth_aligned_img = DepthImage(depth=depth_aligned)
        output_path = os.path.join(output_depth_folder, depth_filename)
        depth_aligned_img.to_pil_image().save(output_path)
    
    if len(scales) == 0:
        raise ValueError("No frames were successfully aligned")
    
    print(f"Successfully aligned {len(scales)} frames")
    print(f"Scale statistics: mean={np.mean(scales):.4f}, std={np.std(scales):.4f}")
    print(f"Shift statistics: mean={np.mean(shifts):.4f}, std={np.std(shifts):.4f}")
    
    params_path = os.path.join(output_depth_folder, "alignment_params.npz")
    np.savez(params_path, scales=np.array(scales), shifts=np.array(shifts))
    print(f"Alignment parameters saved to {params_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Align depth images to SMPL rendered depth images.")
    parser.add_argument("--depth_folder", type=str, required=True)
    parser.add_argument("--smpl_depth_folder", type=str, required=True)
    parser.add_argument("--output_depth_folder", type=str, required=True)
    parser.add_argument("--masks_folder", type=str, required=True)
    parser.add_argument("--smpl_masks_folder", type=str, default=None)
    
    args = parser.parse_args()
    align_depth_to_smpl(
        depth_folder=args.depth_folder,
        smpl_depth_folder=args.smpl_depth_folder,
        output_depth_folder=args.output_depth_folder,
        masks_folder=args.masks_folder,
        smpl_masks_folder=args.smpl_masks_folder
    )
