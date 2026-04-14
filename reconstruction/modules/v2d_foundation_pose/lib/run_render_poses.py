"""
GPU-batched mesh overlay renderer using nvdiffrast.

Renders a mesh at every pose in poses_dir and composites over the matching
background frames from frames_dir.  All poses in a batch are rasterised in a
single nvdiffrast call, so throughput scales with GPU parallelism rather than
frame count.

Supports both UV-textured meshes and vertex-coloured meshes (detected
automatically via make_mesh_tensors).
"""
import argparse
import logging
import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch

from v2d.common.datatypes import CameraIntrinsics, Transform3d
from v2d.foundation_pose.lib.fp_utils import nvdiffrast_render, make_mesh_tensors
from v2d.mesh.lib.mesh import Mesh

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_render_poses(
    mesh_path: str,
    poses_dir: str,
    frames_dir: str,
    intrinsics_path: str,
    output_dir: str,
    batch_size: int = 32,
    use_light: bool = True,
) -> None:
    """
    Render mesh overlays for every pose in poses_dir using nvdiffrast (GPU-batched).

    Args:
        mesh_path:      Mesh to render (GLB, OBJ, etc.).
        poses_dir:      Directory of per-frame Transform3d JSON files.
        frames_dir:     Directory of per-frame background PNG images (000000.png, ...).
        intrinsics_path: Camera intrinsics JSON.
        output_dir:     Destination for composited overlay PNGs.
        batch_size:     Number of poses to rasterise in one GPU call.
                        Reduce if you hit VRAM limits. Default 32.
        use_light:      If True, apply front-facing diffuse lighting.
                        If False, render raw texture/vertex colours. Default True.
    """
    intrinsics = CameraIntrinsics.load(intrinsics_path)
    K  = intrinsics.to_matrix()
    H, W = intrinsics.height, intrinsics.width

    # Upload mesh to GPU once
    trimesh_mesh = Mesh.load(mesh_path).to_trimesh()
    mesh_tensors = make_mesh_tensors(trimesh_mesh)
    logger.info(
        f"Mesh loaded: {len(trimesh_mesh.vertices)} verts, "
        f"{'UV-textured' if 'tex' in mesh_tensors else 'vertex-coloured'}"
    )

    # Enumerate poses
    pose_files   = sorted(f for f in os.listdir(poses_dir) if f.endswith('.json'))
    frame_indices = [int(os.path.splitext(f)[0]) for f in pose_files]
    logger.info(f"Rendering {len(pose_files)} frames (batch_size={batch_size})")

    glctx = dr.RasterizeCudaContext()
    os.makedirs(output_dir, exist_ok=True)

    for chunk_start in range(0, len(pose_files), batch_size):
        chunk_files   = pose_files[chunk_start : chunk_start + batch_size]
        chunk_indices = frame_indices[chunk_start : chunk_start + batch_size]

        # Stack (N, 4, 4) pose batch
        ob_in_cams = torch.stack([
            torch.as_tensor(
                Transform3d.load(os.path.join(poses_dir, f)).to_matrix(),
                device='cuda', dtype=torch.float,
            )
            for f in chunk_files
        ])

        # Single GPU render call for the whole batch
        with torch.no_grad():
            colors, depths, _ = nvdiffrast_render(
                K=K, H=H, W=W,
                ob_in_cams=ob_in_cams,
                glctx=glctx,
                mesh_tensors=mesh_tensors,
                use_light=use_light,
            )

        # colors: (N, H, W, 3) float32 [0, 1], background pixels = 0
        # depths: (N, H, W) float32, background pixels = 0
        colors_np = (colors.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        depths_np = depths.cpu().numpy()

        for frame_idx, color_np, depth_np in zip(chunk_indices, colors_np, depths_np):
            bg_path = os.path.join(frames_dir, f"{frame_idx:06d}.png")
            if os.path.exists(bg_path):
                bg = cv2.cvtColor(cv2.imread(bg_path), cv2.COLOR_BGR2RGB)
                if bg.shape[:2] != (H, W):
                    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
            else:
                bg = np.zeros((H, W, 3), dtype=np.uint8)

            # Alpha from depth: 1 where mesh was rasterised, 0 elsewhere
            alpha      = (depth_np > 0.001).astype(np.float32)[..., None]
            composited = (
                color_np.astype(np.float32) * alpha
                + bg.astype(np.float32) * (1.0 - alpha)
            ).clip(0, 255).astype(np.uint8)

            out_path = os.path.join(output_dir, f"{frame_idx:06d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(composited, cv2.COLOR_RGB2BGR))

        logger.info(f"  Batch {chunk_indices[0]:06d}–{chunk_indices[-1]:06d} done")

    logger.info(f"Rendered {len(pose_files)} overlays → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-batched mesh overlay renderer")
    parser.add_argument("--mesh_path",       required=True)
    parser.add_argument("--poses_dir",       required=True)
    parser.add_argument("--frames_dir",      required=True)
    parser.add_argument("--intrinsics_path", required=True)
    parser.add_argument("--output_dir",      required=True)
    parser.add_argument("--batch_size",      type=int,  default=32)
    parser.add_argument("--use_light",       action="store_true", default=True)
    parser.add_argument("--no_light",        dest="use_light", action="store_false")
    args = parser.parse_args()
    run_render_poses(
        args.mesh_path,
        args.poses_dir,
        args.frames_dir,
        args.intrinsics_path,
        args.output_dir,
        batch_size=args.batch_size,
        use_light=args.use_light,
    )
