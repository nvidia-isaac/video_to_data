"""
Render SMPL depth images from SMPL parameter estimates.
"""
import os
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

from v2d.datatypes import DepthImage, CameraIntrinsics, Mask

from v2d.nlf.lib.smpl_paths import get_smpl_model_root


def render_smpl_depth(
    smpl_params_path: str,
    intrinsics_path: str,
    output_depth_folder: str,
    output_mask_folder: str,
    weights_dir: str,
    device: str = "cuda"
):
    """Render SMPL depth images from SMPL parameter estimates."""
    os.makedirs(output_depth_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)
    
    with h5py.File(smpl_params_path, 'r') as f:
        poses = f['poses'][:]
        betas = f['betas'][:]
        transls = f['transls'][:]
        gender = f['gender'][()].decode('utf-8') if isinstance(f['gender'][()], bytes) else f['gender'][()]
        model_type = f['model_type'][()].decode('utf-8') if isinstance(f['model_type'][()], bytes) else f['model_type'][()]
        frames = [frame.decode('utf-8') if isinstance(frame, bytes) else frame for frame in f['frames'][:]]
    
    num_frames = len(frames)
    print(f"Rendering depth for {num_frames} frames")
    
    with open(intrinsics_path, 'r') as f:
        intrinsics_dict = json.load(f)
    intrinsics = CameraIntrinsics.from_dict(intrinsics_dict)
    
    model_root = get_smpl_model_root(model_type, weights_dir)
    if gender == 'neutral' and model_type == 'smplh':
        gender = 'male'
    
    body_model = BodyModel(model_type, gender, model_root=model_root).to(device)
    
    faces = body_model.faces
    if hasattr(faces, 'cpu'):
        faces_np = faces.cpu().numpy()
    elif hasattr(faces, 'numpy'):
        faces_np = faces.numpy()
    else:
        faces_np = faces
    
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    
    camera = pyrender.IntrinsicsCamera(
        fx=intrinsics.fx, fy=intrinsics.fy,
        cx=intrinsics.cx, cy=intrinsics.cy,
        znear=0.001, zfar=100.0
    )
    camera_pose = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    scene.add(camera, pose=camera_pose)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)
    
    renderer = pyrender.OffscreenRenderer(intrinsics.width, intrinsics.height)
    
    with torch.no_grad():
        poses_t = torch.from_numpy(poses).to(device).float()
        betas_t = torch.from_numpy(betas).to(device).float()
        transls_t = torch.from_numpy(transls).to(device).float()
        
        output = body_model(pose_rotvecs=poses_t, shape_betas=betas_t, trans=transls_t)
        vertices = output['vertices'].cpu().numpy()
    
    for i in tqdm(range(num_frames), desc="Rendering SMPL depth"):
        frame_str = frames[i]
        frame_idx = int(frame_str)
        
        mesh = trimesh.Trimesh(vertices[i], faces_np)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=[0.8, 0.3, 0.3, 1.0]
        )
        render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        
        mesh_node = scene.add(render_mesh)
        
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        scene.remove_node(mesh_node)
        
        depth = depth.astype(np.float32)
        
        mask = Mask(mask=(depth > 0.001).astype(np.uint8))
        depth[~mask.mask] = 0.0
        
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
    parser.add_argument("--smpl_params_path", type=str, required=True)
    parser.add_argument("--intrinsics_path", type=str, required=True)
    parser.add_argument("--output_depth_folder", type=str, required=True)
    parser.add_argument("--output_mask_folder", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=True)
    
    args = parser.parse_args()
    render_smpl_depth(
        smpl_params_path=args.smpl_params_path,
        intrinsics_path=args.intrinsics_path,
        output_depth_folder=args.output_depth_folder,
        output_mask_folder=args.output_mask_folder,
        weights_dir=args.weights_dir
    )
