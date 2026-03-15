"""
SAM3D image to mesh processing function.
Can be called directly from command line or imported as a function.
"""
import torch.hub
torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: None

from v2d.common.datatypes import Transform3d, CameraIntrinsics
from v2d.sam3d.lib.inference_pipeline_modified import InferencePipelinePointMap
import os
import argparse
import numpy as np
from PIL import Image
import json
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

_pipeline = None

def _get_pipeline(weights_dir: str):
    global _pipeline
    if _pipeline is None:
        possible_config_paths = [
            os.path.join(weights_dir, "hf-download", "checkpoints", "pipeline.yaml"),
            os.path.join(weights_dir, "checkpoints", "pipeline.yaml"),
            os.path.join(weights_dir, "pipeline.yaml"),
        ]

        config_file = None
        for path in possible_config_paths:
            if os.path.exists(path):
                config_file = path
                break

        if config_file is None:
            files_in_dir = []
            if os.path.isdir(weights_dir):
                for root, dirs, files in os.walk(weights_dir):
                    if 'pipeline.yaml' in files:
                        files_in_dir.append(os.path.relpath(os.path.join(root, 'pipeline.yaml'), weights_dir))

            raise FileNotFoundError(
                f"SAM3D config file (pipeline.yaml) not found in {weights_dir}\n"
                f"Tried: {possible_config_paths}\n"
                f"Found pipeline.yaml at: {files_in_dir if files_in_dir else 'none'}\n"
                f"Please download checkpoints using: python -m v2d.sam3d.docker.run_download_weights"
            )

        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"
        config.compile_model = False
        config.workspace_dir = os.path.dirname(config_file)
        config._target_ = "v2d.sam3d.lib.inference_pipeline_modified.InferencePipelinePointMap"

        hf_home = os.environ.get("HF_HOME", os.path.join(weights_dir, "hf_home"))
        moge_model_path = None

        moge_cache_paths = [
            os.path.join(hf_home, "hub", "models--Ruicheng--moge-vitl", "snapshots"),
            os.path.join(weights_dir, "hf_home", "hub", "models--Ruicheng--moge-vitl", "snapshots"),
        ]

        for cache_path in moge_cache_paths:
            if os.path.exists(cache_path):
                snapshots = [d for d in os.listdir(cache_path) if os.path.isdir(os.path.join(cache_path, d))]
                if snapshots:
                    moge_model_path = os.path.join(cache_path, snapshots[0], "model.pt")
                    if os.path.exists(moge_model_path):
                        break

        if moge_model_path and os.path.exists(moge_model_path):
            if hasattr(config, 'depth_model') and hasattr(config.depth_model, 'model'):
                config.depth_model.model.pretrained_model_name_or_path = moge_model_path
                print(f"Using local MoGE model: {moge_model_path}")
        else:
            print(f"Warning: Local MoGE model not found, will try to download from HuggingFace")
            print(f"Searched in: {moge_cache_paths}")

        print(f"Initializing SAM3D pipeline with target: {config._target_}")
        try:
            _pipeline = instantiate(config)
            print(f"SAM3D pipeline initialized successfully")
        except Exception as e:
            print(f"Error instantiating pipeline: {e}")
            print(f"Config keys: {list(config.keys())}")
            print(f"Config _target_: {config.get('_target_', 'NOT SET')}")
            raise
    return _pipeline

def _to_opencv_transform(rotation_wxyz: list, translation: list, scale: list) -> Transform3d:
    """
    Convert SAM3D's raw rotation/translation to an OpenCV-convention Transform3d.

    SAM3D internally applies two coordinate changes before projecting:
      1. flip_matrix F: Y-up canonical → Z-forward  (F^T = [[1,0,0],[0,0,-1],[0,1,0]])
      2. Nxy = diag(-1,-1,1): negate X and Y for the final projection

    The net object-to-OpenCV-camera rotation is:
        R_net = Nxy @ R_q^T @ F^T
    and the net translation is:
        t_net = Nxy @ t
    """
    w, x, y, z = rotation_wxyz
    R_q = Rotation.from_quat([x, y, z, w]).as_matrix()  # scipy uses [x,y,z,w]

    F_T = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    Nxy = np.diag([-1.0, -1.0, 1.0])

    R_net = Nxy @ R_q.T @ F_T
    t_net = Nxy @ np.array(translation, dtype=float)

    q_xyzw = Rotation.from_matrix(R_net).as_quat()  # [x,y,z,w]
    q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

    return Transform3d(
        rotation=q_wxyz,
        translation=t_net.tolist(),
        scale=scale,
    )


def _merge_mask_to_rgba(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Merge mask into image alpha channel"""
    mask = mask.astype(np.uint8) * 255
    mask = mask[..., None]
    return np.concatenate([image[..., :3], mask], axis=-1)

def image_to_mesh(image_path: str, mask_path: str, mesh_path: str, transform_path: str, intrinsics_path: str,
                 weights_dir: str,
                 seed: int = None,
                 stage1_only: bool = False,
                 with_mesh_postprocess: bool = False,
                 with_texture_baking: bool = False,
                 with_layout_postprocess: bool = False,
                 use_vertex_color: bool = True,
                 stage1_inference_steps: int = None):
    """Process an image with mask to generate 3D mesh and save outputs to files."""
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    image_array = np.asarray(image)
    mask_array = np.asarray(mask) != 0

    image_height, image_width = image_array.shape[:2]
    mask_array = mask_array.astype(bool) if mask_array.dtype != bool else mask_array
    image_rgba = _merge_mask_to_rgba(image_array, mask_array)

    pipeline = _get_pipeline(weights_dir)
    output = pipeline.run(
        image_rgba,
        None,
        seed,
        stage1_only=stage1_only,
        with_mesh_postprocess=with_mesh_postprocess,
        with_texture_baking=with_texture_baking,
        with_layout_postprocess=with_layout_postprocess,
        use_vertex_color=use_vertex_color,
        stage1_inference_steps=stage1_inference_steps
    )

    mesh_scene = output['glb']

    transform = _to_opencv_transform(
        rotation_wxyz=output['rotation'][0].tolist(),
        translation=output['translation'][0].tolist(),
        scale=output['scale'][0].tolist(),
    )

    intrinsics = CameraIntrinsics(
        fx=output['intrinsics'][0, 0].item() * image_width,
        fy=output['intrinsics'][1, 1].item() * image_height,
        cx=output['intrinsics'][0, 2].item() * image_width,
        cy=output['intrinsics'][1, 2].item() * image_height,
        width=image_width,
        height=image_height
    )

    with open(mesh_path, "wb") as f:
        f.write(mesh_scene.export(file_type='glb'))

    with open(transform_path, "w") as f:
        json.dump(transform.to_dict(), f, indent=4)

    with open(intrinsics_path, "w") as f:
        json.dump(intrinsics.to_dict(), f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process image to mesh using SAM3D")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to input mask")
    parser.add_argument("--mesh_path", type=str, required=True, help="Output path for mesh GLB")
    parser.add_argument("--transform_path", type=str, required=True, help="Output path for transform JSON")
    parser.add_argument("--intrinsics_path", type=str, required=True, help="Output path for intrinsics JSON")
    parser.add_argument("--weights_dir", type=str, required=True, help="Path to weights directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--stage1_only", action="store_true", help="Only run stage 1")
    parser.add_argument("--with_mesh_postprocess", action="store_true", help="Enable mesh postprocessing")
    parser.add_argument("--with_texture_baking", action="store_true", help="Enable texture baking")
    parser.add_argument("--with_layout_postprocess", action="store_true", help="Enable layout postprocessing")
    parser.add_argument("--use_vertex_color", action="store_true", default=True, help="Use vertex color")
    parser.add_argument("--stage1_inference_steps", type=int, default=None, help="Stage 1 inference steps")

    args = parser.parse_args()
    image_to_mesh(
        args.image_path,
        args.mask_path,
        args.mesh_path,
        args.transform_path,
        args.intrinsics_path,
        weights_dir=args.weights_dir,
        seed=args.seed,
        stage1_only=args.stage1_only,
        with_mesh_postprocess=args.with_mesh_postprocess,
        with_texture_baking=args.with_texture_baking,
        with_layout_postprocess=args.with_layout_postprocess,
        use_vertex_color=args.use_vertex_color,
        stage1_inference_steps=args.stage1_inference_steps
    )
