"""
Align NLF SMPL predictions to depth images using ICP.
Based on cari4d_internal/prep/align_nlf2unidepth.py
"""
import os
import sys
import cv2
import numpy as np
import torch
import h5py
import json
import trimesh
import open3d as o3d
from tqdm import tqdm
from PIL import Image
from typing import Optional

# Try to import FoundationPose Utils for GPU-accelerated depth filtering
try:
    # Try module path first
    from modules.foundationpose.FoundationPose.Utils import erode_depth, bilateral_filter_depth, depth2xyzmap
    USE_FP_UTILS = True
except ImportError:
    try:
        # Try direct import (if FoundationPose is in path)
        from FoundationPose.Utils import erode_depth, bilateral_filter_depth, depth2xyzmap
        USE_FP_UTILS = True
    except ImportError:
        try:
            # Try adding FoundationPose to path
            fp_path = os.path.join(os.path.dirname(__file__), '../../foundationpose/FoundationPose')
            if os.path.exists(fp_path):
                sys.path.insert(0, fp_path)
                from Utils import erode_depth, bilateral_filter_depth, depth2xyzmap
                USE_FP_UTILS = True
            else:
                USE_FP_UTILS = False
        except ImportError:
            print("Warning: Could not import FoundationPose Utils, using CPU fallback")
            USE_FP_UTILS = False

from modules.common.datatypes import DepthImage, CameraIntrinsics, Mask
from modules.nlf.icp_utils import translation_only_icp_torch
from smplfitter.pt import BodyModel, BodyFitter


def depth2xyzmap_simple(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert depth map to xyz point cloud map (CPU version).
    Similar to FoundationPose Utils.depth2xyzmap but simpler.
    """
    invalid_mask = (depth < 0.001)
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map


def erode_depth_cpu(depth: np.ndarray, radius: int = 2, 
                    depth_diff_thres: float = 0.001, 
                    ratio_thres: float = 0.8, 
                    zfar: float = 100.0) -> np.ndarray:
    """
    CPU version of depth erosion.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    depth = np.asarray(depth, dtype=np.float32, order='C')
    H, W = depth.shape
    k = 2 * radius + 1

    # Pad for windows
    depth_pad = np.pad(depth, radius, mode='constant', constant_values=0.0)

    # Sliding windows (H,W,k,k)
    Dp = sliding_window_view(depth_pad, (k, k))

    # In-bounds neighbor count
    ones = np.ones((H, W), dtype=np.uint8)
    ones_pad = np.pad(ones, radius, mode='constant', constant_values=0)
    totals = sliding_window_view(ones_pad, (k, k)).sum(axis=(2, 3)).astype(np.float32)

    center = depth[..., None, None]
    bad = ((Dp < 0.001) | (Dp >= zfar) | (np.abs(Dp - center) > depth_diff_thres))
    bad_count = bad.sum(axis=(2, 3)).astype(np.float32)

    ratio = bad_count / np.maximum(totals, 1.0)
    center_invalid = (depth < 0.001) | (depth >= zfar)
    out = np.where(center_invalid | (ratio > ratio_thres), 0.0, depth).astype(np.float32)
    return out


def bilateral_filter_depth_cpu(depth: np.ndarray,
                               radius: int = 2,
                               zfar: float = 100.0,
                               sigmaD: float = 2.0,
                               sigmaR: float = 100000.0) -> np.ndarray:
    """
    CPU version of bilateral filter for depth.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    
    depth = np.asarray(depth, dtype=np.float32, order='C')
    H, W = depth.shape
    k = 2 * radius + 1

    # Pad image and masks
    depth_pad = np.pad(depth, radius, mode='constant', constant_values=0.0)
    valid = (depth >= 0.001) & (depth < zfar)
    valid_pad = np.pad(valid.astype(np.uint8), radius, mode='constant', constant_values=0)

    # Sliding windows
    Dp = sliding_window_view(depth_pad, (k, k))
    Vp = sliding_window_view(valid_pad, (k, k)).astype(bool)

    # Local valid neighbor count and mean
    num_valid = Vp.sum(axis=(2, 3)).astype(np.float32)
    sum_valid = (Dp * Vp).sum(axis=(2, 3), dtype=np.float32)
    mean_valid = sum_valid / np.maximum(num_valid, 1.0)

    # Spatial Gaussian
    ys = np.arange(-radius, radius + 1, dtype=np.float32)
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
    spatial = np.exp(-(grid_x**2 + grid_y**2) / (2.0 * sigmaD * sigmaD)).astype(np.float32)

    # Gate by local valid mean
    gate = Vp & (np.abs(Dp - mean_valid[..., None, None]) < 0.01)

    # Range Gaussian around center depth
    center = depth[..., None, None]
    range_w = np.exp(-((center - Dp) ** 2) / (2.0 * sigmaR * sigmaR)).astype(np.float32)

    # Weights and normalization
    weights = range_w * spatial[None, None, ...]
    weights = np.where(gate, weights, 0.0).astype(np.float32)
    sum_w = weights.sum(axis=(2, 3), dtype=np.float32)

    # Weighted sum
    num = (weights * Dp).sum(axis=(2, 3), dtype=np.float32)
    out = np.zeros((H, W), dtype=np.float32)
    valid_out = (sum_w > 0.0) & (num_valid > 0.0)
    out[valid_out] = (num[valid_out] / sum_w[valid_out]).astype(np.float32)
    return out


def align_nlf_to_depth(
    smpl_results_path: str,
    depth_folder: str,
    masks_dir: str,
    intrinsics_path: str,
    output_path: str,
    device: str = "cuda"
):
    """
    Align NLF SMPL predictions to depth images using ICP.
    
    Args:
        smpl_results_path: Path to NLF SMPL results h5 file
        depth_folder: Folder containing depth images (000000.png, 000001.png, ...)
        masks_dir: Directory containing person masks (000000.png, 000001.png, ...)
        intrinsics_path: Path to camera intrinsics JSON file
        output_path: Path to save aligned SMPL results h5 file
        device: Device to use ('cuda' or 'cpu')
    """
    # Load intrinsics
    with open(intrinsics_path, 'r') as f:
        intrinsics_dict = json.load(f)
    intrinsics = CameraIntrinsics(**intrinsics_dict)
    K = intrinsics.to_matrix()
    
    # Load NLF results
    print(f"Loading NLF results from {smpl_results_path}...")
    with h5py.File(smpl_results_path, 'r') as f:
        poses = f['poses'][:]
        betas = f['betas'][:]
        transls = f['transls'][:]
        gender = f['gender'][()].decode('utf-8') if isinstance(f['gender'][()], bytes) else f['gender'][()]
        model_type = f['model_type'][()].decode('utf-8') if isinstance(f['model_type'][()], bytes) else f['model_type'][()]
        frames = [frame.decode('utf-8') if isinstance(frame, bytes) else frame for frame in f['frames'][:]]
    
    num_frames = len(frames)
    print(f"Found {num_frames} frames")
    
    # Initialize SMPL model
    smpl_model_root = os.environ.get('SMPL_MODEL_ROOT', os.path.join(os.environ.get('DATA_DIR', '/data'), 'nlf/smpl_models'))
    model_root = os.path.join(smpl_model_root, model_type)
    
    if gender == 'neutral' and model_type == 'smplh':
        gender = 'male'  # smplh doesn't have neutral
    
    body_model = BodyModel(model_type, gender, model_root=model_root).to(device)
    fitter = BodyFitter(body_model).to(device)
    
    # Get SMPL faces for mesh sampling
    faces = body_model.faces
    if hasattr(faces, 'cpu'):
        faces_np = faces.cpu().numpy()
    elif hasattr(faces, 'numpy'):
        faces_np = faces.numpy()
    else:
        faces_np = faces
    
    # Process each frame
    verts_aligned = []
    valid_frames = []
    
    print("Aligning NLF to depth for each frame...")
    for i in tqdm(range(num_frames), desc="Processing frames"):
        frame_str = frames[i]
        frame_idx = int(frame_str)
        
        # Load depth
        depth_path = os.path.join(depth_folder, f"{frame_idx:06d}.png")
        if not os.path.exists(depth_path):
            print(f"Warning: Depth not found for frame {frame_idx}, skipping")
            continue
        
        # Load mask
        mask_path = os.path.join(masks_dir, f"{frame_idx:06d}.png")
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for frame {frame_idx}, skipping")
            continue
        
        # Load and process depth
        depth_img = DepthImage.load(depth_path)
        depth = depth_img.depth  # Already in meters from DepthImage
        
        # Ensure depth is in reasonable range (0.1m to 50m)
        depth = np.clip(depth, 0.1, 50.0)
        
        # Load mask
        mask = Mask.load(mask_path).mask > 0
        
        # Filter depth
        if USE_FP_UTILS:
            depth_tensor = torch.as_tensor(depth, device=device, dtype=torch.float)
            depth_tensor = erode_depth(depth_tensor, radius=2, device=device)
            depth_tensor = bilateral_filter_depth(depth_tensor, radius=2, device=device)
            depth_filtered = depth_tensor.cpu().numpy()
        else:
            depth_filtered = erode_depth_cpu(depth, radius=2)
            depth_filtered = bilateral_filter_depth_cpu(depth_filtered, radius=2)
        
        # Convert depth to point cloud
        if USE_FP_UTILS:
            dmap_xyz = depth2xyzmap(depth_filtered, K)
        else:
            dmap_xyz = depth2xyzmap_simple(depth_filtered, K)
        
        # Extract human points from mask
        # dmap_xyz is (H, W, 3), mask is (H, W)
        pts_hum = dmap_xyz[mask > 0]
        if pts_hum.ndim == 2 and pts_hum.shape[1] == 3:
            pass  # Already correct shape
        else:
            pts_hum = pts_hum.reshape((-1, 3))
        if len(pts_hum) < 100:
            print(f"Warning: Too few valid depth points ({len(pts_hum)}) for frame {frame_idx}, skipping")
            continue
        
        # Get NLF SMPL vertices for this frame
        pose_frame = torch.from_numpy(poses[i:i+1]).to(device).float()
        beta_frame = torch.from_numpy(betas[i:i+1]).to(device).float()
        transl_frame = torch.from_numpy(transls[i:i+1]).to(device).float()
        
        with torch.no_grad():
            output = body_model(
                pose_rotvecs=pose_frame,
                shape_betas=beta_frame,
                trans=transl_frame
            )
            verts_nlf = output['vertices'][0].cpu().numpy()
        
        # Sample points from SMPL mesh
        smpl_mesh = trimesh.Trimesh(vertices=verts_nlf, faces=faces_np, process=False)
        pts_nlf_sample = smpl_mesh.sample(8000)
        
        # Initial z-translation alignment
        z_pts_nlf = np.median(pts_nlf_sample[:, 2])
        z_pts_hum_median = np.median(pts_hum[:, 2])
        mat = np.eye(4)
        mat[:3, 3] = [0, 0, z_pts_hum_median - z_pts_nlf]
        pts_nlf = np.matmul(pts_nlf_sample, mat[:3, :3].T) + mat[:3, 3]
        
        # Run translation-only ICP
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_nlf))
        tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_hum))
        mat_icp = translation_only_icp_torch(src, tgt, voxel_size=0.01, max_iters=[25, 10, 5])
        mat = np.matmul(mat_icp, mat)
        
        # Estimate scale based on z-depth
        z_nlf_orig = np.mean(verts_nlf[:, 2])
        z_nlf_new = mat[:3, 3][2] + z_nlf_orig
        scale = z_nlf_new / z_nlf_orig if z_nlf_orig > 0.001 else 1.0
        
        # Apply scale and redo alignment
        mat_scale = np.eye(4)
        np.fill_diagonal(mat_scale, scale)
        mat_scale[3, 3] = 1
        mat_scale[:3, 3] = [0, 0, z_pts_hum_median - z_pts_nlf * scale]
        
        src_scaled = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            np.matmul(pts_nlf_sample, mat_scale[:3, :3].T) + mat_scale[:3, 3]
        ))
        mat_icp2 = translation_only_icp_torch(src_scaled, tgt, voxel_size=0.01)
        mat = np.matmul(mat_icp2, mat_scale)
        
        # Apply transformation to vertices
        verts_nlf_align = np.matmul(verts_nlf, mat[:3, :3].T) + mat[:3, 3]
        
        # Validate alignment
        z = np.mean(verts_nlf_align[:, 2])
        if z < 1.0:
            print(f"Warning: z is too small {z} on frame {frame_idx}, using original NLF")
            verts_nlf_align = verts_nlf
        
        verts_aligned.append(verts_nlf_align)
        valid_frames.append(i)
    
    if len(verts_aligned) == 0:
        raise ValueError("No valid frames found for alignment")
    
    print(f"Successfully aligned {len(verts_aligned)} frames")
    
    # Stack aligned vertices and refit SMPL parameters
    print("Refitting SMPL parameters to aligned vertices...")
    verts_aligned = np.stack(verts_aligned, 0)
    verts_tensor = torch.from_numpy(verts_aligned).to(device).float()
    
    fit_res = fitter.fit(
        verts_tensor,
        num_iter=3,
        beta_regularizer=1,
        requested_keys=['shape_betas', 'trans', 'pose_rotvecs']
    )
    
    # Save aligned results
    print(f"Saving aligned results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Only save valid frames
        valid_poses = fit_res['pose_rotvecs'].cpu().numpy()
        valid_betas = fit_res['shape_betas'].cpu().numpy()
        valid_transls = fit_res['trans'].cpu().numpy()
        valid_frames_list = [frames[i] for i in valid_frames]
        
        f.create_dataset('poses', data=valid_poses)
        f.create_dataset('betas', data=valid_betas)
        f.create_dataset('transls', data=valid_transls)
        f.create_dataset('gender', data=gender.encode('utf-8'))
        f.create_dataset('model_type', data=model_type.encode('utf-8'))
        f.create_dataset('frames', data=[f.encode('utf-8') for f in valid_frames_list])
        
        # Compute and save aligned vertices
        with torch.no_grad():
            output = body_model(
                pose_rotvecs=fit_res['pose_rotvecs'],
                shape_betas=fit_res['shape_betas'],
                trans=fit_res['trans']
            )
            aligned_vertices = output['vertices'].cpu().numpy()
        
        f.create_dataset('vertices', data=aligned_vertices)
        f.create_dataset('faces', data=faces_np)
    
    print(f"Aligned SMPL results saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Align NLF SMPL predictions to depth images using ICP.")
    parser.add_argument("--smpl_results_path", type=str, required=True, help="Path to NLF SMPL results h5 file")
    parser.add_argument("--depth_folder", type=str, required=True, help="Folder containing depth images")
    parser.add_argument("--masks_dir", type=str, required=True, help="Directory containing person masks")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Path to camera intrinsics JSON")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save aligned SMPL results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    align_nlf_to_depth(
        smpl_results_path=args.smpl_results_path,
        depth_folder=args.depth_folder,
        masks_dir=args.masks_dir,
        intrinsics_path=args.intrinsics_path,
        output_path=args.output_path,
        device=args.device
    )

