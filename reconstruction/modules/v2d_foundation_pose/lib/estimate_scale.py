"""
Given a reconstructed textured mesh, estimate the scale using foundation pose
"""
import sys, os
import torch, json
import trimesh
import numpy as np
from tqdm import tqdm
import cv2
import os.path as osp
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV

_FP_DIR = osp.join(osp.dirname(osp.abspath(__file__)), 'FoundationPose')
sys.path.insert(0, _FP_DIR)

from estimater import FoundationPose
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor
import nvdiffrast.torch as dr
from v2d.foundation_pose.lib import fp_utils as Utils
from v2d.foundation_pose.lib.chamfer_dist_np import chamfer_distance
from v2d.datatypes import CameraIntrinsics, DepthImage, Mask

def fp_scale_estimator(
    mesh_path: str,
    rgb_path: str,
    depth_path: str,
    mask_path: str,
    intrinsics_path: str,
    transform_path: str,
    output_transform_path: str,
    debug_dir: str = None,
    num_levels: int = 3,
    num_samples_per_level: int = 10,
    level_size: float = 2.0
):
    """Use FoundationPose to estimate the rough scale by trying multiple scale candidates.
    
    Uses a hierarchical multi-level search:
    1. Initialize with input transform scale
    2. For each level, search num_samples_per_level candidates centered around current best
    3. Range is [current_best / level_size, current_best * level_size]
    4. Repeat for N levels
    """
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    with open(intrinsics_path, 'r') as f:
        intrinsics_dict = json.load(f)
    intrinsics = CameraIntrinsics.from_dict(intrinsics_dict)
    camera_K = intrinsics.to_matrix()

    with open(transform_path, 'r') as f:
        transform_data = json.load(f)
    
    initial_scale = transform_data.get('scale', [1.0, 1.0, 1.0])
    if isinstance(initial_scale, list):
        initial_scale = np.mean(initial_scale)
    current_best_scale = float(initial_scale)
    
    color = cv2.imread(rgb_path)
    if color is None:
        raise FileNotFoundError(f"Could not load RGB image from {rgb_path}")
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    
    depth_img = DepthImage.load(depth_path)
    depth = depth_img.depth
    
    mask_img = Mask.load(mask_path)
    mask_o = mask_img.mask
    if mask_o.max() <= 1:
        mask_o = (mask_o * 255).astype(np.uint8)

    depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth_tensor = Utils.erode_depth(depth_tensor, radius=2, device='cuda')
    depth_tensor = Utils.bilateral_filter_depth(depth_tensor, radius=2, device='cuda')
    depth_filtered = depth_tensor.cpu().numpy()
    
    xyz_map = Utils.depth2xyzmap(depth_filtered, camera_K)
    obj_pts = xyz_map[mask_o > 127].reshape((-1, 3))

    mesh_raw = trimesh.load(mesh_path, process=False)
    if hasattr(mesh_raw, 'geometry'):
        mesh_raw = trimesh.util.concatenate([g for g in mesh_raw.geometry.values()])
    
    samples = mesh_raw.sample(10000)

    if debug_dir is None:
        debug_dir = "/tmp/foundationpose_debug"
    os.makedirs(debug_dir, exist_ok=True)

    print(f"Initializing scale search with transform scale: {current_best_scale:.4f}")
    print(f"Using {num_levels} levels with {num_samples_per_level} samples per level (level_size={level_size})")
    
    best_pose = None
    best_chamfer = float('inf')
    
    for level in range(num_levels):
        scale_min = current_best_scale / level_size
        scale_max = current_best_scale * level_size
        
        scale_candidates = np.linspace(scale_min, scale_max, num_samples_per_level)
        
        print(f"\nLevel {level + 1}/{num_levels}: Searching {num_samples_per_level} candidates in range [{scale_min:.4f}, {scale_max:.4f}]")
        
        level_best_scale = current_best_scale
        level_best_chamfer = best_chamfer
        level_best_pose = best_pose
        
        for scale in tqdm(scale_candidates, desc=f"Level {level + 1}"):
            mesh_t = mesh_raw.copy()
            mesh_t.vertices = mesh_t.vertices * scale
            
            est = FoundationPose(
                model_pts=mesh_t.vertices, 
                model_normals=mesh_t.vertex_normals, 
                mesh=mesh_t, 
                scorer=scorer,
                refiner=refiner, 
                debug_dir=debug_dir, 
                debug=0, 
                glctx=glctx
            )
            
            pose = est.register(
                K=camera_K, 
                rgb=color, 
                depth=depth_filtered, 
                ob_mask=mask_o.astype(bool),
                iteration=5
            )
            
            samples_t = samples * scale
            samples_cam = np.matmul(samples_t, pose[:3, :3].T) + pose[:3, 3]
            
            cd = chamfer_distance(obj_pts, samples_cam)
            
            if cd < level_best_chamfer:
                level_best_scale = scale
                level_best_chamfer = cd
                level_best_pose = pose
        
        current_best_scale = level_best_scale
        best_chamfer = level_best_chamfer
        best_pose = level_best_pose
        
        print(f"Level {level + 1} best: scale={current_best_scale:.4f}, Chamfer={best_chamfer:.4f}")

    best_scale = current_best_scale
    print(f"\nFinal best scale: {best_scale:.4f} (Chamfer: {best_chamfer:.4f})")

    refined_transform = transform_data.copy()
    refined_transform['scale'] = [best_scale, best_scale, best_scale]
    refined_transform['translation'] = best_pose[:3, 3].tolist()

    with open(output_transform_path, "w") as f:
        json.dump(refined_transform, f, indent=4)
    print(f"Saved refined transform to {output_transform_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Estimate mesh scale using FoundationPose registration and Chamfer distance")
    parser.add_argument("--mesh", required=True, help="Path to mesh GLB/OBJ")
    parser.add_argument("--rgb", required=True, help="Path to RGB image")
    parser.add_argument("--depth", required=True, help="Path to depth image")
    parser.add_argument("--mask", required=True, help="Path to object mask")
    parser.add_argument("--intrinsics", required=True, help="Path to camera intrinsics JSON")
    parser.add_argument("--transform", required=True, help="Path to original transform JSON")
    parser.add_argument("--output-transform", required=True, help="Path to save refined transform JSON")
    parser.add_argument("--debug-dir", help="Optional directory for debug visualizations")
    parser.add_argument("--num-levels", type=int, default=3, help="Number of hierarchical search levels")
    parser.add_argument("--num-samples-per-level", type=int, default=10, help="Number of scale candidates per level")
    parser.add_argument("--level-size", type=float, default=2.0, help="Search range multiplier per level")

    args = parser.parse_args()
    
    fp_scale_estimator(
        mesh_path=args.mesh,
        rgb_path=args.rgb,
        depth_path=args.depth,
        mask_path=args.mask,
        intrinsics_path=args.intrinsics,
        transform_path=args.transform,
        output_transform_path=args.output_transform,
        debug_dir=args.debug_dir,
        num_levels=args.num_levels,
        num_samples_per_level=args.num_samples_per_level,
        level_size=args.level_size
    )
