import torch
import numpy as np
import os
import pickle as pkl
from smplfitter.pt import BodyModel, BodyFitter
from modules.nlf.lib_smpl.const import SMPL_MODEL_ROOT
from modules.nlf.datatypes import NlfResult
from typing import List

import h5py

def vertices_to_smpl(vertices_path: str, gender: str, model_type: str = "smplh", output_path: str = None, device: str = "cuda") -> NlfResult:
    """
    Fits SMPL/SMPLH parameters to 3D vertices.
    Input vertices: (T, 6890, 3) in meters, loaded from HDF5.
    """
    # Load vertices from HDF5
    with h5py.File(vertices_path, 'r') as f:
        vertices = f['vertices'][:]

    if vertices.size == 0:
        raise ValueError("Vertices array is empty")

    # Initialize BodyFitter
    # Note: smplfitter's BodyModel takes 'smpl' or 'smplh'
    body_model = BodyModel(model_type, gender, model_root=SMPL_MODEL_ROOT).to(device)
    fitter = BodyFitter(body_model, num_betas=10).to(device)

    verts_tensor = torch.from_numpy(vertices).to(device).float()
    
    # Run fitting
    # num_iter=3 is used in the original run_nlf_sepK.py
    fit_res = fitter.fit(
        verts_tensor, 
        num_iter=3,
        beta_regularizer=1,
        requested_keys=['shape_betas', 'trans', 'vertices', 'pose_rotvecs']
    )

    # Extract results
    poses = fit_res['pose_rotvecs'].cpu().numpy()
    betas = fit_res['shape_betas'].cpu().numpy()
    transls = fit_res['trans'].cpu().numpy()
    
    # Generate frame names
    frames = [f"{i:06d}" for i in range(len(vertices))]

    res = NlfResult(
        poses=poses,
        betas=betas,
        transls=transls,
        gender=gender,
        frames=frames,
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
            # Frames as fixed-length strings
            frames_encoded = [f.encode('utf-8') for f in res.frames]
            f.create_dataset('frames', data=frames_encoded)
        print(f"SMPL parameters saved to {output_path} (HDF5)")

    return res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fit SMPL/SMPLH parameters to 3D vertices.")
    parser.add_argument("--vertices_path", type=str, required=True, help="Path to HDF5 file containing vertices")
    parser.add_argument("--gender", type=str, required=True, choices=["male", "female", "neutral"])
    parser.add_argument("--model_type", type=str, default="smplh", choices=["smpl", "smplh"])
    parser.add_argument("--output_path", type=str, required=True)
    
    args = parser.parse_args()
    
    vertices_to_smpl(args.vertices_path, args.gender, args.model_type, args.output_path)

