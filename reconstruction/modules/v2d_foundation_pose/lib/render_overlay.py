import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import numpy as np
import torch
import json
import trimesh
import pyrender
import argparse
from v2d.datatypes import CameraIntrinsics

def render_overlay(
    video_path: str,
    poses_dir: str,
    mesh_path: str,
    camera_intrinsics_path: str,
    output_dir: str,
    device: str = "cuda"
):
    """Renders object overlay on video frames."""
    os.makedirs(output_dir, exist_ok=True)

    scene = trimesh.load(mesh_path, force='scene')
    mesh = scene.geometry[list(scene.geometry.keys())[0]]

    with open(camera_intrinsics_path, "r") as f:
        camera_intrinsics_dict = json.load(f)
    camera_intrinsics = CameraIntrinsics.from_dict(camera_intrinsics_dict)
    
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    
    camera = pyrender.IntrinsicsCamera(
        fx=camera_intrinsics.fx, fy=camera_intrinsics.fy,
        cx=camera_intrinsics.cx, cy=camera_intrinsics.cy
    )
    camera_pose_base = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    camera_node = scene.add(camera, pose=camera_pose_base)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose_base)

    renderer = pyrender.OffscreenRenderer(camera_intrinsics.width, camera_intrinsics.height)

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.2,
        alphaMode='OPAQUE',
        baseColorFactor=[0.3, 0.8, 0.3, 1.0]
    )
    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        pose_path = os.path.join(poses_dir, f"{i:06d}.json")
        if not os.path.exists(pose_path):
            cv2.imwrite(os.path.join(output_dir, f"{i:06d}.jpg"), frame)
            continue

        with open(pose_path, "r") as f:
            pose = np.array(json.load(f))

        mesh_node = scene.add(render_mesh, pose=pose)
        
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
            print(f"Rendered frame {i}/{num_frames}")

    cap.release()
    renderer.delete()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render FoundationPose overlay on video.")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--poses_dir", type=str, required=True)
    parser.add_argument("--mesh_path", type=str, required=True)
    parser.add_argument("--camera_intrinsics_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    args = parser.parse_args()
    render_overlay(
        args.video_path, 
        args.poses_dir, 
        args.mesh_path, 
        args.camera_intrinsics_path, 
        args.output_dir
    )
