"""
FoundationPose video to poses processing function.
Can be called directly from command line or imported as a function.
"""
from v2d.datatypes import CameraIntrinsics, DepthImage, Mask
import os
import sys
import argparse
import numpy as np
import trimesh
import torch
import cv2
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_FP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FoundationPose')
sys.path.insert(0, _FP_DIR)

from estimater import FoundationPose
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor
import nvdiffrast.torch as dr
from Utils import draw_posed_3d_box, draw_xyz_axis


def log_gpu_memory(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        logger.info(f"GPU Memory [{label}]: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


_scorer = None
_refiner = None
_glctx = None

def _get_models():
    global _scorer, _refiner, _glctx
    if _scorer is None or _refiner is None or _glctx is None:
        print("Initializing FoundationPose models...")
        log_gpu_memory("Before model init")
        _scorer = ScorePredictor()
        log_gpu_memory("After Scorer init")
        _refiner = PoseRefinePredictor()
        log_gpu_memory("After Refiner init")
        _glctx = dr.RasterizeCudaContext()
        log_gpu_memory("After GL Context init")
        print("Models initialized successfully")
    return _scorer, _refiner, _glctx

def video_to_poses(video_path: str, depth_folder: str, masks_folder: str, camera_intrinsics_path: str, mesh_path: str, poses_dir: str, reference_frame: int = 0, target_width: int = None, target_height: int = None, debug_dir: str = None):
    """Process a video to track object poses."""
    target_resolution = None
    if target_width and target_height:
        target_resolution = (target_width, target_height)
    
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
    
    scorer, refiner, glctx = _get_models()
    
    log_gpu_memory("Start of run")
    scene = trimesh.load(mesh_path, force='scene')
    mesh = scene.geometry[list(scene.geometry.keys())[0]]
    
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    
    with open(camera_intrinsics_path, "r") as f:
        camera_intrinsics_dict = json.load(f)
    camera_intrinsics = CameraIntrinsics.from_dict(camera_intrinsics_dict)

    with torch.no_grad():
        log_gpu_memory("Before FoundationPose init")
        est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            glctx=glctx,
            debug=0,
            debug_dir='/tmp/foundationpose_debug'
        )
        log_gpu_memory("After FoundationPose init")

        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        K = camera_intrinsics.to_matrix()
        
        if target_resolution is not None:
            target_width, target_height = target_resolution
            scale_x = target_width / orig_width
            scale_y = target_height / orig_height
            
            K_scaled = K.copy()
            K_scaled[0, 0] *= scale_x
            K_scaled[1, 1] *= scale_y
            K_scaled[0, 2] *= scale_x
            K_scaled[1, 2] *= scale_y
            
            print(f"Scaling from {orig_width}x{orig_height} to {target_width}x{target_height}")
            print(f"Scaled K: {K_scaled}")
        else:
            target_width, target_height = orig_width, orig_height
            K_scaled = K

        def get_frame_data(idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                return None, None, None, None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_path = os.path.join(depth_folder, f"{idx:06d}.png")
            mask_path = os.path.join(masks_folder, f"{idx:06d}.png")
            
            depth = None
            if os.path.exists(depth_path):
                depth = DepthImage.load(depth_path).depth
            
            mask = None
            if os.path.exists(mask_path):
                mask = Mask.load(mask_path).mask
            
            if target_resolution is not None:
                rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                if depth is not None:
                    depth = cv2.resize(depth, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
                if mask is not None:
                    mask = cv2.resize(mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            
            return rgb, depth, mask, frame

        def save_pose(frame_idx, pose):
            os.makedirs(poses_dir, exist_ok=True)
            pose_path = os.path.join(poses_dir, f"{frame_idx:06d}.json")
            with open(pose_path, "w") as f:
                json.dump(pose.tolist(), f, indent=4)

        def save_visualization(frame_idx, rgb, pose, K):
            if debug_dir is None:
                return
            vis_img = rgb.copy()
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            vis_img = draw_posed_3d_box(K, vis_img, pose, bbox)
            vis_img = draw_xyz_axis(vis_img, pose, scale=0.05, K=K)
            vis_path = os.path.join(debug_dir, f"{frame_idx:06d}.png")
            cv2.imwrite(vis_path, vis_img)

        print(f"Initializing at reference frame {reference_frame}")
        rgb, depth, mask, _ = get_frame_data(reference_frame)
        if rgb is None or depth is None or mask is None:
            raise RuntimeError(f"Failed to load data for reference frame {reference_frame}")
        
        log_gpu_memory("Before register")
        initial_pose = est.register(K=K_scaled, rgb=rgb, depth=depth, ob_mask=mask, iteration=10)
        log_gpu_memory("After register")

        save_pose(reference_frame, initial_pose)
        save_visualization(reference_frame, rgb, initial_pose, K_scaled)
        print("Initialized at reference frame")
        
        print(f"Tracking forward from {reference_frame + 1} to {num_frames - 1}")
        est.pose_last = torch.as_tensor(initial_pose, device='cuda', dtype=torch.float)
        for frame_idx in range(reference_frame + 1, num_frames):
            rgb, depth, _, _ = get_frame_data(frame_idx)
            if rgb is None or depth is None:
                break
            print(f"Processing forward frame {frame_idx}/{num_frames}")
            pose = est.track_one(rgb=rgb, depth=depth, K=K_scaled, iteration=5)
            save_pose(frame_idx, pose)
            save_visualization(frame_idx, rgb, pose, K_scaled)
            if frame_idx % 10 == 0:
                log_gpu_memory(f"Forward tracking frame {frame_idx}")

        if reference_frame > 0:
            print(f"Tracking backward from {reference_frame - 1} to 0")
            torch.cuda.empty_cache()
            log_gpu_memory("Before backward pass (after empty_cache)")
            est.pose_last = torch.as_tensor(initial_pose, device='cuda', dtype=torch.float)
            for frame_idx in range(reference_frame - 1, -1, -1):
                rgb, depth, _, _ = get_frame_data(frame_idx)
                if rgb is None or depth is None:
                    break
                print(f"Processing backward frame {frame_idx}/{num_frames}")
                pose = est.track_one(rgb=rgb, depth=depth, K=K_scaled, iteration=2)
                save_pose(frame_idx, pose)
                save_visualization(frame_idx, rgb, pose, K_scaled)
                if frame_idx % 10 == 0:
                    log_gpu_memory(f"Backward tracking frame {frame_idx}")
        
        cap.release()
        print(f"Completed processing {num_frames} frames")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video to poses using FoundationPose")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--depth_folder", type=str, required=True, help="Folder containing depth images")
    parser.add_argument("--masks_folder", type=str, required=True, help="Folder containing mask images")
    parser.add_argument("--camera_intrinsics_path", type=str, required=True, help="Path to camera intrinsics JSON")
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to mesh file")
    parser.add_argument("--poses_dir", type=str, required=True, help="Output directory for poses")
    parser.add_argument("--reference_frame", type=int, default=0, help="Reference frame index")
    parser.add_argument("--target_width", type=int, default=None, help="Target width for scaling")
    parser.add_argument("--target_height", type=int, default=None, help="Target height for scaling")
    parser.add_argument("--debug_dir", type=str, default=None, help="Directory to save visualization images")
    
    args = parser.parse_args()
    video_to_poses(
        args.video_path,
        args.depth_folder,
        args.masks_folder,
        args.camera_intrinsics_path,
        args.mesh_path,
        args.poses_dir,
        reference_frame=args.reference_frame,
        target_width=args.target_width,
        target_height=args.target_height,
        debug_dir=args.debug_dir
    )
