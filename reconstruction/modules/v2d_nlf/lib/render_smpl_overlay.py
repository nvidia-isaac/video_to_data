import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import numpy as np
import torch
import h5py
import json
from smplfitter.pt import BodyModel
from v2d.datatypes import CameraIntrinsics
import trimesh
import pyrender

from v2d.nlf.lib.smpl_paths import get_smpl_model_root

def render_smpl_overlay(
    video_path: str,
    smpl_params_path: str,
    intrinsics_path: str,
    output_dir: str,
    weights_dir: str,
    device: str = "cuda"
):
    """Renders SMPL overlay on video frames."""
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(smpl_params_path, 'r') as f:
        poses = f['poses'][:]
        betas = f['betas'][:]
        transls = f['transls'][:]
        gender = f['gender'][()].decode('utf-8')
        model_type = f['model_type'][()].decode('utf-8')

    with open(intrinsics_path, 'r') as f:
        intrinsics_dict = json.load(f)
    intrinsics = CameraIntrinsics.from_dict(intrinsics_dict)
    K = intrinsics.to_matrix()

    model_root = get_smpl_model_root(model_type, weights_dir)
    body_model = BodyModel(model_type, gender, model_root=model_root).to(device)

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    
    camera = pyrender.IntrinsicsCamera(
        fx=intrinsics.fx, fy=intrinsics.fy,
        cx=intrinsics.cx, cy=intrinsics.cy
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
        faces = body_model.faces
        if hasattr(faces, 'cpu'):
            faces = faces.cpu().numpy()
        elif hasattr(faces, 'numpy'):
            faces = faces.numpy()

    for i in range(min(len(vertices), num_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        
        mesh = trimesh.Trimesh(vertices[i], faces)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=[0.8, 0.3, 0.3, 1.0]
        )
        render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        
        mesh_node = scene.add(render_mesh)
        
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        scene.remove_node(mesh_node)
        
        color = color.astype(np.float32) / 255.0
        valid_mask = color[:, :, 3:4] > 0
        
        frame_float = frame.astype(np.float32) / 255.0
        render_bgr = color[:, :, :3][:, :, ::-1]
        
        alpha = 0.7
        overlay = frame_float.copy()
        overlay[valid_mask.squeeze()] = (
            alpha * render_bgr[valid_mask.squeeze()] + 
            (1 - alpha) * frame_float[valid_mask.squeeze()]
        )
        
        out_frame = (overlay * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{i:06d}.jpg"), out_frame)
        
        if i % 10 == 0:
            print(f"Rendered frame {i}/{len(vertices)}")

    cap.release()
    renderer.delete()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render SMPL overlay on video.")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--smpl_params_path", type=str, required=True)
    parser.add_argument("--intrinsics_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weights_dir", type=str, required=True)
    
    args = parser.parse_args()
    render_smpl_overlay(
        args.video_path, args.smpl_params_path, args.intrinsics_path,
        args.output_dir, args.weights_dir
    )
