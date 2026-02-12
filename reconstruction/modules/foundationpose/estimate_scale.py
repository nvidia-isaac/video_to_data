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

# Add FoundationPose to path
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), 'FoundationPose'))

from estimater import FoundationPose
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor
import nvdiffrast.torch as dr
import Utils
from chamfer_dist_np import chamfer_distance
from modules.common.datatypes import CameraIntrinsics, DepthImage, Mask

def fp_scale_estimator(
    mesh_path: str,
    rgb_path: str,
    depth_path: str,
    mask_path: str,
    intrinsics_path: str,
    transform_path: str,
    output_transform_path: str,
    debug_dir: str = None
):
    """Use FoundationPose to estimate the rough scale by trying multiple scale candidates."""
    # 1. Initialize models
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    # 2. Load data
    with open(intrinsics_path, 'r') as f:
        intrinsics_dict = json.load(f)
    intrinsics = CameraIntrinsics.from_dict(intrinsics_dict)
    camera_K = intrinsics.to_matrix()

    with open(transform_path, 'r') as f:
        transform_data = json.load(f)
    
    color = cv2.imread(rgb_path)
    if color is None:
        raise FileNotFoundError(f"Could not load RGB image from {rgb_path}")
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    
    depth_img = DepthImage.load(depth_path)
    depth = depth_img.depth # in meters
    
    mask_img = Mask.load(mask_path)
    mask_o = mask_img.mask
    if mask_o.max() <= 1:
        mask_o = (mask_o * 255).astype(np.uint8)

    # 3. Preprocess depth
    depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth_tensor = Utils.erode_depth(depth_tensor, radius=2, device='cuda')
    depth_tensor = Utils.bilateral_filter_depth(depth_tensor, radius=2, device='cuda')
    depth_filtered = depth_tensor.cpu().numpy()
    
    xyz_map = Utils.depth2xyzmap(depth_filtered, camera_K)
    obj_pts = xyz_map[mask_o > 127].reshape((-1, 3))

    # 4. Load mesh
    mesh_raw = trimesh.load(mesh_path, process=False)
    if hasattr(mesh_raw, 'geometry'):
        mesh_raw = trimesh.util.concatenate([g for g in mesh_raw.geometry.values()])
    
    # 5. Scale estimation loop
    scale_candidates = np.linspace(0.75, 1.25, 30)
    samples = mesh_raw.sample(10000)
    chamfs = []
    pose_results = []

    if debug_dir is None:
        debug_dir = "/tmp/foundationpose_debug"
    os.makedirs(debug_dir, exist_ok=True)

    print(f"Estimating scale with {len(scale_candidates)} candidates...")
    for scale in tqdm(scale_candidates):
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
        
        # We use register to find the best pose for this specific scale
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
        chamfs.append([scale, cd])
        pose_results.append(pose)

    # 6. Pick best scale
    chamfs = np.array(chamfs)
    best_idx = np.argmin(chamfs[:, 1])
    best_scale = chamfs[best_idx, 0]
    best_pose = pose_results[best_idx]
    
    print(f"Best scale found: {best_scale} (Chamfer: {chamfs[best_idx, 1]:.4f})")

    # 7. Save refined transform
    # The output format should match align_mesh_scale.py's expectation
    refined_transform = transform_data.copy()
    # Note: SAM3D transform has scale as a 3-element list, but here we assume uniform scale
    refined_transform['scale'] = [best_scale, best_scale, best_scale]
    # We also update translation based on the best pose found during registration
    # FoundationPose pose is [R|t] where t is translation in camera space
    refined_transform['translation'] = best_pose[:3, 3].tolist()
    # We should probably also update rotation, but FoundationPose uses 4x4 matrix 
    # while SAM3D uses quaternions. For now, let's keep it simple or convert if needed.
    # Since the goal is scale alignment, we focus on scale and translation.

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

    args = parser.parse_args()
    
    fp_scale_estimator(
        mesh_path=args.mesh,
        rgb_path=args.rgb,
        depth_path=args.depth,
        mask_path=args.mask,
        intrinsics_path=args.intrinsics,
        transform_path=args.transform,
        output_transform_path=args.output_transform,
        debug_dir=args.debug_dir
    )
