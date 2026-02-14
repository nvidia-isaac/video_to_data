import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

import numpy as np
# Patch numpy for chumpy compatibility
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'complex'): np.complex = complex
if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'unicode'): np.unicode = str
if not hasattr(np, 'str'): np.str = str

import os
import cv2
import torch
import torchvision
import json
import h5py
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple
from smplfitter.pt import BodyModel, BodyFitter
from modules.nlf.datatypes import CameraIntrinsics, NlfResult

# Singleton model instance
_nlf_model = None

def _get_nlf_model():
    global _nlf_model
    if _nlf_model is None:
        weights_path = os.environ.get("NLF_WEIGHTS_PATH", "modules/nlf/data/weights/nlf_l_multi_0.3.2.torchscript")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"NLF weights not found at {weights_path}. Please run download.sh or place weights manually.")
        _nlf_model = torch.jit.load(weights_path).cuda().eval()
    return _nlf_model

def masks2bbox(masks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box from a list of masks."""
    all_idx = []
    for mask in masks:
        idx = np.where(mask > 127)
        if len(idx[0]) > 0:
            all_idx.append(np.stack(idx, axis=1))
    
    if len(all_idx) == 0:
        return np.array([0, 0]), np.array([0, 0])
    
    all_idx = np.concatenate(all_idx, axis=0)
    bmin = np.min(all_idx, axis=0)[::-1]  # (x, y)
    bmax = np.max(all_idx, axis=0)[::-1]  # (x, y)
    return bmin, bmax

def video_to_smpl(
    video_path: str, 
    masks_dir: str, 
    intrinsics_path: str, 
    gender: str, 
    model_type: str = "smplh",
    output_path: str = None,
    chunk_size: int = 32,
    device: str = "cuda"
) -> NlfResult:
    """
    End-to-end NLF: Video + Masks -> SMPL Parameters (all in memory).
    """
    # 1. Load intrinsics
    with open(intrinsics_path, 'r') as f:
        intrinsics_dict = json.load(f)
    intrinsics = CameraIntrinsics(**intrinsics_dict)
    K = intrinsics.to_matrix()
    K_tensor = torch.from_numpy(K).to(device).float()[None]

    # 2. Localization: Video -> 3D Vertices
    model = _get_nlf_model()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_vertices = []
    
    for i in tqdm(range(0, frame_count, chunk_size), desc="Running NLF Localization"):
        images_chunk = []
        bboxes_chunk = []

        for j in range(i, min(i + chunk_size, frame_count)):
            ret, frame = cap.read()
            if not ret:
                break
            
            mask_path = os.path.join(masks_dir, f"{j:06d}.png")
            if not os.path.exists(mask_path):
                continue

            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3: mask = mask[:, :, 0]
            if mask.max() <= 1: mask = (mask * 255).astype(np.uint8)

            bmin, bmax = masks2bbox([mask])
            if np.all(bmin == 0) and np.all(bmax == 0):
                continue

            bbox = np.concatenate([bmin, bmax - bmin, [1.01]])  # xywhc
            images_chunk.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            bboxes_chunk.append(torch.from_numpy(bbox)[None].to(device).float())

        if not images_chunk:
            continue

        image_tensor = torch.from_numpy(np.stack(images_chunk)).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            pred = model.detect_smpl_batched(
                image_tensor, 
                extra_boxes=bboxes_chunk, 
                detector_threshold=1.0, 
                suppress_implausible_poses=False, 
                intrinsic_matrix=K_tensor
            )
        verts = torch.cat(pred['vertices3d'], dim=0).cpu().numpy() / 1000.0
        all_vertices.append(verts)

    cap.release()
    
    if not all_vertices:
        raise ValueError("No valid person masks found in video.")
    
    vertices = np.concatenate(all_vertices, axis=0)

    # 3. Fitting: 3D Vertices -> SMPL Parameters
    smpl_model_root = os.environ.get('SMPL_MODEL_ROOT', os.path.join(os.environ.get('DATA_DIR', '/data'), 'nlf/smpl_models'))
    model_root = os.path.join(smpl_model_root, model_type)
    
    if gender == 'neutral' and model_type == 'smplh':
        gender = 'male' # smplh doesn't have neutral
    
    body_model = BodyModel(model_type, gender, model_root=model_root).to(device)
    fitter = BodyFitter(body_model).to(device)
    verts_tensor = torch.from_numpy(vertices).to(device).float()
    
    fit_res = fitter.fit(
        verts_tensor, 
        num_iter=3,
        beta_regularizer=1,
        share_beta=True,  # Share betas across all frames to ensure constant shape/scale
        requested_keys=['shape_betas', 'trans', 'pose_rotvecs']
    )

    # Compute vertices from fitted parameters using body_model
    with torch.no_grad():
        output = body_model(
            pose_rotvecs=fit_res['pose_rotvecs'],
            shape_betas=fit_res['shape_betas'],
            trans=fit_res['trans']
        )
        fitted_vertices = output['vertices'].cpu().numpy()

    res = NlfResult(
        poses=fit_res['pose_rotvecs'].cpu().numpy(),
        betas=fit_res['shape_betas'].cpu().numpy(),
        transls=fit_res['trans'].cpu().numpy(),
        gender=gender,
        frames=[f"{i:06d}" for i in range(len(vertices))],
        model_type=model_type
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('poses', data=res.poses)
            f.create_dataset('betas', data=res.betas)
            f.create_dataset('transls', data=res.transls)
            f.create_dataset('gender', data=res.gender.encode('utf-8'))
            f.create_dataset('model_type', data=res.model_type.encode('utf-8'))
            f.create_dataset('frames', data=[f.encode('utf-8') for f in res.frames])
            # Save vertices and faces for easier visualization
            f.create_dataset('vertices', data=fitted_vertices)
            faces = body_model.faces
            if hasattr(faces, 'cpu'):
                faces = faces.cpu().numpy()
            elif hasattr(faces, 'numpy'):
                faces = faces.numpy()
            f.create_dataset('faces', data=faces)
        print(f"Results saved to {output_path}")

    return res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="End-to-end NLF: Video + Masks -> SMPL Parameters.")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--intrinsics_path", type=str, required=True)
    parser.add_argument("--gender", type=str, required=True, choices=["male", "female", "neutral"])
    parser.add_argument("--model_type", type=str, default="smplh", choices=["smpl", "smplh"])
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=32)
    
    args = parser.parse_args()
    video_to_smpl(args.video_path, args.masks_dir, args.intrinsics_path, args.gender, args.model_type, args.output_path, args.chunk_size)
