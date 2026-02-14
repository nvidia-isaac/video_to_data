"""
Render SMPL depth images from SMPL parameter estimates.
"""
import os
# Set EGL as the backend for pyrender before importing it
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import torch
import h5py
import json
from tqdm import tqdm
import trimesh
import pyrender
import cv2
from smplfitter.pt import BodyModel

from modules.common.datatypes import DepthImage, CameraIntrinsics, Mask

SMPL_MODEL_ROOT = os.environ.get('SMPL_MODEL_ROOT', os.path.join(os.environ.get('DATA_DIR', '/data'), 'nlf/smpl_models'))


def render_smpl_depth(
    smpl_params_path: str,
    intrinsics_path: str,
    output_depth_folder: str,
    output_mask_folder: str,
    device: str = "cuda"
):
    """
    Render SMPL depth images from SMPL parameter estimates.
    
    Args:
        smpl_params_path: Path to SMPL parameters h5 file
        intrinsics_path: Path to camera intrinsics JSON file
        output_depth_folder: Folder to save rendered depth images
        output_mask_folder: Folder to save rendered masks (uses standard naming: 000000.png, etc.)
        device: Device to use ('cuda' or 'cpu')
    """
    os.makedirs(output_depth_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)
    
    # Load SMPL parameters
    with h5py.File(smpl_params_path, 'r') as f:
        poses = f['poses'][:]
        betas = f['betas'][:]
        transls = f['transls'][:]
        gender = f['gender'][()].decode('utf-8') if isinstance(f['gender'][()], bytes) else f['gender'][()]
        model_type = f['model_type'][()].decode('utf-8') if isinstance(f['model_type'][()], bytes) else f['model_type'][()]
        frames = [frame.decode('utf-8') if isinstance(frame, bytes) else frame for frame in f['frames'][:]]
    
    num_frames = len(frames)
    print(f"Rendering depth for {num_frames} frames")
    
    # Load intrinsics
    with open(intrinsics_path, 'r') as f:
        intrinsics_dict = json.load(f)
    intrinsics = CameraIntrinsics.from_dict(intrinsics_dict)
    
    # Initialize BodyModel
    model_root = os.path.join(SMPL_MODEL_ROOT, model_type)
    if gender == 'neutral' and model_type == 'smplh':
        gender = 'male'  # smplh doesn't have neutral
    
    body_model = BodyModel(model_type, gender, model_root=model_root).to(device)
    
    # Get faces
    faces = body_model.faces
    if hasattr(faces, 'cpu'):
        faces_np = faces.cpu().numpy()
    elif hasattr(faces, 'numpy'):
        faces_np = faces.numpy()
    else:
        faces_np = faces
    
    # Setup pyrender scene
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    
    # Camera
    camera = pyrender.IntrinsicsCamera(
        fx=intrinsics.fx, fy=intrinsics.fy,
        cx=intrinsics.cx, cy=intrinsics.cy,
        znear=0.001, zfar=100.0
    )
    # Convert OpenCV camera coordinate system to OpenGL (flip Y and Z)
    camera_pose = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    scene.add(camera, pose=camera_pose)
    
    # Light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)
    
    renderer = pyrender.OffscreenRenderer(intrinsics.width, intrinsics.height)
    
    # Generate vertices for all frames
    with torch.no_grad():
        poses_t = torch.from_numpy(poses).to(device).float()
        betas_t = torch.from_numpy(betas).to(device).float()
        transls_t = torch.from_numpy(transls).to(device).float()
        
        output = body_model(pose_rotvecs=poses_t, shape_betas=betas_t, trans=transls_t)
        vertices = output['vertices'].cpu().numpy()
    
    # Render depth for each frame
    for i in tqdm(range(num_frames), desc="Rendering SMPL depth"):
        frame_str = frames[i]
        frame_idx = int(frame_str)
        
        # Create mesh for this frame
        mesh = trimesh.Trimesh(vertices[i], faces_np)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=[0.8, 0.3, 0.3, 1.0]
        )
        render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        
        # Add mesh to scene
        mesh_node = scene.add(render_mesh)
        
        # Render (returns color and depth)
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        scene.remove_node(mesh_node)
        
        depth = depth.astype(np.float32)
        
        mask = Mask(mask=(depth > 0.001).astype(np.uint8))
        
        depth[~mask.mask] = 0.0
        
        # Save depth image
        depth_img = DepthImage(depth=depth)
        output_path = os.path.join(output_depth_folder, f"{frame_idx:06d}.png")
        depth_img.to_pil_image().save(output_path)
        mask_output_path = os.path.join(output_mask_folder, f"{frame_idx:06d}.png")
        mask.to_pil_image().save(mask_output_path)
        
    renderer.delete()
    print(f"Rendered {num_frames} depth images and masks to {output_depth_folder} and {output_mask_folder}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render SMPL depth images from SMPL estimates.")
    parser.add_argument("--smpl_params_path", type=str, required=True,
                       help="Path to SMPL parameters h5 file")
    parser.add_argument("--intrinsics_path", type=str, required=True,
                       help="Path to camera intrinsics JSON file")
    parser.add_argument("--output_depth_folder", type=str, required=True,
                       help="Folder to save rendered depth images")
    parser.add_argument("--output_mask_folder", type=str, required=True,
                       help="Folder to save rendered masks (uses standard naming: 000000.png, etc.)")
    
    args = parser.parse_args()
    render_smpl_depth(
        smpl_params_path=args.smpl_params_path,
        intrinsics_path=args.intrinsics_path,
        output_depth_folder=args.output_depth_folder,
        output_mask_folder=args.output_mask_folder
    )

