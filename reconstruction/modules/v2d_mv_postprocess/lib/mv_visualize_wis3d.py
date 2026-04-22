from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from v2d.mv.rig import RigConfig
from v2d.mv.vis.wis3d_helper import Wis3DScene

HUMAN_MESH_COLOR = np.array([102, 230, 179], dtype=np.uint8)


def visualize_wis3d(
    output_dir: Path,
    object_mesh: trimesh.Trimesh,
    object_poses: np.ndarray,
    human_vertices: np.ndarray,
    human_faces: np.ndarray,
    cam_extrinsics: dict[str, np.ndarray] | None = None,
):
    """Generate a Wis3D visualization with per-frame object + human meshes.

    Args:
        output_dir: Where to write the Wis3D output.
        object_mesh: Object mesh in canonical frame.
        object_poses: (N, 4, 4) per-frame object-to-world poses.
        human_vertices: (N, V, 3) human vertices in world frame.
        human_faces: (F, 3) human mesh face indices.
        cam_extrinsics: Optional dict of {cam_name: (4, 4) T_world_from_camera}
            for camera frustum visualization.
    """
    n_frames = min(len(object_poses), len(human_vertices))
    human_colors = np.tile(HUMAN_MESH_COLOR, (human_vertices.shape[1], 1))

    scene = Wis3DScene(output_dir, name="hoi_scene")

    for i in tqdm(range(n_frames), desc="Building Wis3D scene"):
        scene.set_frame(i)

        scene.add_axes("world", np.eye(4))

        obj_mesh_i = object_mesh.copy()
        obj_mesh_i.apply_transform(object_poses[i])
        scene.add_mesh("object", obj_mesh_i)

        human_mesh_i = trimesh.Trimesh(
            vertices=human_vertices[i],
            faces=human_faces,
            vertex_colors=human_colors,
            process=False,
        )
        scene.add_mesh("human", human_mesh_i)

        if cam_extrinsics:
            for cam_name, T in cam_extrinsics.items():
                scene.add_axes(f"cam_{cam_name}", T, scale=0.5)

    print(f"Saved Wis3D visualization to {output_dir}")


def visualize_wis3d_from_config(cfg):
    rig = RigConfig(cfg.rig_config, camera_params_path=cfg.camera_params_path)

    object_mesh = trimesh.load(cfg.object_mesh_path, process=False, force='mesh')
    object_poses = np.load(cfg.object_pose_path)

    mhr_mesh = torch.load(cfg.mhr_mesh_mv_path, weights_only=False, map_location="cpu")

    human_vertices = mhr_mesh["pred_vertices"].cpu().numpy()
    human_faces = mhr_mesh["faces"].cpu().numpy()

    cam_extrinsics = {}
    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)
        cam_extrinsics[cam.name] = cam.param.T

    visualize_wis3d(
        output_dir=Path(cfg.output_dir),
        object_mesh=object_mesh,
        object_poses=object_poses,
        human_vertices=human_vertices,
        human_faces=human_faces,
        cam_extrinsics=cam_extrinsics,
    )


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Generate Wis3D visualization from config")
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--object_mesh_path", type=str, required=True)
    parser.add_argument("--object_pose_dir", type=str, required=True)
    parser.add_argument("--human_pose_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(Path(__file__).parent / "mv_visualize_wis3d.yaml"),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    overrides = {k: v for k, v in vars(args).items() if k != "config_path" and v is not None}
    cfg = OmegaConf.merge(cfg, overrides)
    visualize_wis3d_from_config(cfg)
