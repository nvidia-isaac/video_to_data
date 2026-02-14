"""
Align depth images to SMPL rendered depth images.
Based on cari4d_internal/prep/align_monod2hum.py

This module provides functionality to align monocular depth images to SMPL-rendered depth images,
using robust scale and shift computation to provide absolute depth scale.
"""
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import glob

# Try to import FoundationPose Utils for GPU-accelerated depth filtering
erode_depth = None
bilateral_filter_depth = None
import_error_msgs = []

try:
    from modules.foundationpose.FoundationPose.Utils import erode_depth, bilateral_filter_depth
except ImportError as e:
    import_error_msgs.append(f"modules.foundationpose.FoundationPose.Utils: {str(e)}")
    try:
        from FoundationPose.Utils import erode_depth, bilateral_filter_depth
    except ImportError as e:
        import_error_msgs.append(f"FoundationPose.Utils: {str(e)}")
        try:
            # Try absolute path from /app
            fp_path = '/app/modules/foundationpose/FoundationPose'
            if os.path.exists(fp_path):
                sys.path.insert(0, fp_path)
                from Utils import erode_depth, bilateral_filter_depth
            else:
                # Try relative path
                fp_path = os.path.join(os.path.dirname(__file__), '../../foundationpose/FoundationPose')
                fp_path = os.path.abspath(fp_path)
                if os.path.exists(fp_path):
                    sys.path.insert(0, fp_path)
                    from Utils import erode_depth, bilateral_filter_depth
                else:
                    import_error_msgs.append(f"FoundationPose path not found: /app/modules/foundationpose/FoundationPose or {fp_path}")
        except ImportError as e:
            import_error_msgs.append(f"Utils (from path): {str(e)}")

if erode_depth is None or bilateral_filter_depth is None:
    error_msg = "FoundationPose Utils not found. GPU-accelerated depth filtering is required.\n"
    error_msg += "Import attempts:\n" + "\n".join(f"  - {msg}" for msg in import_error_msgs)
    error_msg += "\n\nNote: FoundationPose Utils requires warp-lang. Ensure it's installed in the Docker image."
    raise ImportError(error_msg)

from modules.common.datatypes import DepthImage, Mask


def compute_scale_and_shift_robust(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """
    Robust scale and shift computation as proposed in:
    https://arxiv.org/pdf/1907.01341
    
    Args:
        pred: Predicted depth map (numpy array)
        target: Target depth map (numpy array)
        mask: Boolean mask indicating valid pixels for alignment
    
    Returns:
        scale: Scale factor
        shift: Shift value
    """
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
    """
    Align depth images to SMPL rendered depth images.
    
    This function reads depth images and SMPL depth images from folders,
    computes robust scale and shift for each frame, and outputs aligned depth images.
    
    Args:
        depth_folder: Folder containing input depth images (e.g., 000000.png, 000001.png, ...)
        smpl_depth_folder: Folder containing SMPL rendered depth images (same naming convention)
        output_depth_folder: Folder to save aligned depth images
        masks_folder: Folder containing person masks (from segmentation)
        smpl_masks_folder: Folder containing SMPL rendered masks (if None, looks in smpl_depth_folder)
    """
    device = "cuda"
    os.makedirs(output_depth_folder, exist_ok=True)
    
    # Get all depth files
    depth_files = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
    if len(depth_files) == 0:
        raise ValueError(f"No depth images found in {depth_folder}")
    
    print(f"Found {len(depth_files)} depth images")
    
    scales = []
    shifts = []
    
    for depth_path in tqdm(depth_files, desc="Aligning depth to SMPL"):
        # Get corresponding SMPL depth file
        depth_filename = os.path.basename(depth_path)
        smpl_depth_path = os.path.join(smpl_depth_folder, depth_filename)
        
        if not os.path.exists(smpl_depth_path):
            print(f"Warning: SMPL depth not found for {depth_filename}, skipping")
            continue
        
        # Load depth images
        depth_img = DepthImage.load(depth_path)
        depth = depth_img.depth
        
        smpl_depth_img = DepthImage.load(smpl_depth_path)
        smpl_depth = smpl_depth_img.depth
        
        # Don't clip invalid depths to 0.1 - that creates fake clipped values
        # Instead, set invalid depths to 0 (they'll be filtered out)
        # Only clip upper bound to reasonable range
        depth = np.clip(depth, 0.0, 50.0)
        smpl_depth = np.clip(smpl_depth, 0.0, 50.0)
        # Set very small values to 0 (they're likely invalid)
        depth[depth < 0.001] = 0.0
        smpl_depth[smpl_depth < 0.001] = 0.0
        
        # Filter depths using GPU
        depth_tensor = torch.as_tensor(depth, device=device, dtype=torch.float32)
        depth_tensor = erode_depth(depth_tensor, radius=2, device=device)
        depth_tensor = bilateral_filter_depth(depth_tensor, radius=2, device=device)
        depth_filtered = depth_tensor.cpu().numpy()
        
        smpl_depth_tensor = torch.as_tensor(smpl_depth, device=device, dtype=torch.float32)
        smpl_depth_tensor = erode_depth(smpl_depth_tensor, radius=2, device=device)
        smpl_depth_tensor = bilateral_filter_depth(smpl_depth_tensor, radius=2, device=device)
        smpl_depth_filtered = smpl_depth_tensor.cpu().numpy()
        
        # Load input human mask (from segmentation)
        mask_path = os.path.join(masks_folder, depth_filename)
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {depth_filename}, skipping")
            continue
        
        input_mask = Mask.load(mask_path).mask > 0
        
        # Load SMPL rendered mask (from depth rendering)
        # Mask files use standard naming convention: 000000.png (same as depth files)
        if smpl_masks_folder is None:
            smpl_masks_folder = smpl_depth_folder
        
        smpl_mask_path = os.path.join(smpl_masks_folder, depth_filename)
        if not os.path.exists(smpl_mask_path):
            # Fallback: create mask from depth (for backward compatibility)
            print(f"Warning: SMPL rendered mask not found for {depth_filename} in {smpl_masks_folder}, using depth-based mask")
            smpl_rendered_mask = smpl_depth_filtered > 0.001
        else:
            smpl_rendered_mask = Mask.load(smpl_mask_path).mask > 0
        
        # Create alignment mask: intersection of:
        # 1. Input human mask (from segmentation)
        # 2. SMPL rendered mask (where mesh was actually rendered)
        # 3. Valid depth pixels in both (depth > threshold, excluding clipped values)
        # Exclude clipped values (0.1m is the clipping threshold, so exclude values <= 0.15m)
        MIN_VALID_DEPTH = 0.15  # Exclude depths that are likely clipped
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
        
        # Compute scale and shift
        scale, shift = compute_scale_and_shift_robust(
            depth_filtered, 
            smpl_depth_filtered, 
            alignment_mask
        )
        
        # Compute spread statistics for debugging
        valid_depth = depth_filtered[alignment_mask]
        valid_smpl_depth = smpl_depth_filtered[alignment_mask]
        median_depth = np.median(valid_depth)
        median_smpl_depth = np.median(valid_smpl_depth)
        spread_depth = np.mean(np.abs(valid_depth - median_depth))
        spread_smpl_depth = np.mean(np.abs(valid_smpl_depth - median_smpl_depth))
        
        # Check for suspicious scales (likely indicates a problem)
        if scale < 0.01 or scale > 100.0:
            print(f"[WARNING] Suspicious scale {scale:.6f} for {depth_filename}")
            print(f"  Monocular depth range: [{valid_depth.min():.3f}, {valid_depth.max():.3f}]")
            print(f"  SMPL depth range: [{valid_smpl_depth.min():.3f}, {valid_smpl_depth.max():.3f}]")
            print(f"  Monocular spread (MAD): {spread_depth:.6f}m")
            print(f"  SMPL spread (MAD): {spread_smpl_depth:.6f}m")
            print(f"  Scale = SMPL_spread / Monocular_spread = {spread_smpl_depth:.6f} / {spread_depth:.6f}")
            print(f"  Mask stats: input_mask={np.sum(input_mask)}, smpl_rendered_mask={np.sum(smpl_rendered_mask)}, alignment={np.sum(alignment_mask)}")
        
        scales.append(scale)
        shifts.append(shift)
        
        # Debug: Print statistics for first frame
        if len(scales) == 1:
            # Count depth value distribution
            smpl_all_depths = smpl_depth_filtered[smpl_rendered_mask & (smpl_depth_filtered > 0.001)]
            clipped_in_mask = np.sum((smpl_depth_filtered > 0.001) & (smpl_depth_filtered <= MIN_VALID_DEPTH) & smpl_rendered_mask)
            
            print(f"[DEBUG] Alignment frame {depth_filename}:")
            print(f"  Masks: input_mask={np.sum(input_mask)}, smpl_rendered_mask={np.sum(smpl_rendered_mask)}, alignment={np.sum(alignment_mask)}")
            print(f"  SMPL depth in rendered mask: total={len(smpl_all_depths)}, clipped(<=0.15m)={clipped_in_mask}, valid(>{MIN_VALID_DEPTH}m)={len(valid_smpl_depth)}")
            if len(smpl_all_depths) > 0:
                print(f"  SMPL depth (all in mask): min={smpl_all_depths.min():.3f}m, max={smpl_all_depths.max():.3f}m, "
                      f"median={np.median(smpl_all_depths):.3f}m")
            print(f"  Monocular depth (aligned): min={valid_depth.min():.3f}m, max={valid_depth.max():.3f}m, "
                  f"median={median_depth:.3f}m, mean={valid_depth.mean():.3f}m, spread(MAD)={spread_depth:.6f}m")
            print(f"  SMPL depth (aligned): min={valid_smpl_depth.min():.3f}m, max={valid_smpl_depth.max():.3f}m, "
                  f"median={median_smpl_depth:.3f}m, mean={valid_smpl_depth.mean():.3f}m, spread(MAD)={spread_smpl_depth:.6f}m")
            print(f"  Computed scale: {scale:.6f} (from spread ratio: {spread_smpl_depth:.6f} / {spread_depth:.6f})")
            print(f"  Computed shift: {shift:.6f}")
        
        # Apply alignment to original depth (not filtered)
        depth_aligned = depth.copy()
        valid_mask = depth > 0.001
        depth_aligned[valid_mask] = depth_aligned[valid_mask] * scale + shift
        
        # Clip to reasonable range
        depth_aligned = np.clip(depth_aligned, 0.1, 50.0)
        
        # Save aligned depth
        depth_aligned_img = DepthImage(depth=depth_aligned)
        output_path = os.path.join(output_depth_folder, depth_filename)
        depth_aligned_img.to_pil_image().save(output_path)
    
    if len(scales) == 0:
        raise ValueError("No frames were successfully aligned")
    
    print(f"Successfully aligned {len(scales)} frames")
    print(f"Scale statistics: mean={np.mean(scales):.4f}, std={np.std(scales):.4f}")
    print(f"Shift statistics: mean={np.mean(shifts):.4f}, std={np.std(shifts):.4f}")
    
    # Save alignment parameters
    params_path = os.path.join(output_depth_folder, "alignment_params.npz")
    np.savez(params_path, scales=np.array(scales), shifts=np.array(shifts))
    print(f"Alignment parameters saved to {params_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Align depth images to SMPL rendered depth images.")
    parser.add_argument("--depth_folder", type=str, required=True, 
                       help="Folder containing input depth images")
    parser.add_argument("--smpl_depth_folder", type=str, required=True,
                       help="Folder containing SMPL rendered depth images")
    parser.add_argument("--output_depth_folder", type=str, required=True,
                       help="Folder to save aligned depth images")
    parser.add_argument("--masks_folder", type=str, required=True,
                       help="Folder containing person masks (from segmentation)")
    parser.add_argument("--smpl_masks_folder", type=str, default=None,
                       help="Folder containing SMPL rendered masks (if not provided, looks in smpl_depth_folder)")
    
    args = parser.parse_args()
    align_depth_to_smpl(
        depth_folder=args.depth_folder,
        smpl_depth_folder=args.smpl_depth_folder,
        output_depth_folder=args.output_depth_folder,
        masks_folder=args.masks_folder,
        smpl_masks_folder=args.smpl_masks_folder
    )
