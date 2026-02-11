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
import pickle as pkl
from modules.nlf.datatypes import CameraIntrinsics, NlfResult
from modules.nlf.functions.video_to_vertices import video_to_vertices
from modules.nlf.functions.vertices_to_smpl import vertices_to_smpl

def video_to_smpl(
    video_path: str, 
    masks_dir: str, 
    intrinsics_path: str, 
    gender: str, 
    model_type: str = "smplh",
    output_path: str = None
) -> NlfResult:
    """
    End-to-end orchestrator for NLF: Video + Masks -> SMPL Parameters.
    """
    # Create temporary path for vertices
    temp_vertices_path = output_path.replace(".h5", "_vertices.h5") if output_path else "temp_vertices.h5"

    # 1. Localization: Video -> 3D Vertices (via HDF5)
    video_to_vertices(video_path, masks_dir, intrinsics_path, output_path=temp_vertices_path)
    
    # 2. Fitting: 3D Vertices -> SMPL Parameters (via HDF5)
    result = vertices_to_smpl(temp_vertices_path, gender, model_type, output_path=output_path)
    
    # Cleanup temp file if it wasn't explicitly requested
    if not output_path and os.path.exists(temp_vertices_path):
        os.remove(temp_vertices_path)

    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="End-to-end NLF: Video + Masks -> SMPL Parameters.")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--intrinsics_path", type=str, required=True)
    parser.add_argument("--gender", type=str, required=True, choices=["male", "female", "neutral"])
    parser.add_argument("--model_type", type=str, default="smplh", choices=["smpl", "smplh"])
    parser.add_argument("--output_path", type=str, required=True)
    
    args = parser.parse_args()
    
    video_to_smpl(args.video_path, args.masks_dir, args.intrinsics_path, args.gender, args.model_type, args.output_path)

