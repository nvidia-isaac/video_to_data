from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from tqdm import tqdm

from v2d.mv.rig import RigConfig
from v2d.mv.vis.wis3d_helper import Wis3DScene

HUMAN_MESH_COLOR = np.array([102, 230, 179], dtype=np.uint8)
GROUND_COLOR_A = np.array([200, 200, 200], dtype=np.uint8)
GROUND_COLOR_B = np.array([100, 100, 100], dtype=np.uint8)


def _make_ground_plane_meshes(
    plane: np.ndarray,
    center: np.ndarray,
    size: float = 4.0,
    square_size: float = 0.5,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Create two checkerboard meshes (light + dark) on the plane.

    Returns separate meshes for light and dark squares so that each has
    uniform vertex color (avoids interpolation artifacts in Wis3D).
    """
    normal = plane[:3] / np.linalg.norm(plane[:3])
    d_norm = plane[3] / np.linalg.norm(plane[:3])

    proj_center = center - (np.dot(normal, center) + d_norm) * normal

    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    u_axis = np.cross(normal, arbitrary)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)

    n_squares = int(np.ceil(size / square_size))
    half = n_squares * square_size / 2.0

    groups: dict[int, tuple[list, list, int]] = {
        0: ([], [], 0),  # light
        1: ([], [], 0),  # dark
    }

    for i in range(n_squares):
        for j in range(n_squares):
            u0 = -half + i * square_size
            v0 = -half + j * square_size
            u1 = u0 + square_size
            v1 = v0 + square_size

            corners = np.array([
                proj_center + u0 * u_axis + v0 * v_axis,
                proj_center + u1 * u_axis + v0 * v_axis,
                proj_center + u1 * u_axis + v1 * v_axis,
                proj_center + u0 * u_axis + v1 * v_axis,
            ])

            key = (i + j) % 2
            verts_list, faces_list, offset = groups[key]
            verts_list.append(corners)
            faces_list.append(np.array([
                [offset, offset + 1, offset + 2],
                [offset, offset + 2, offset + 3],
            ]))
            groups[key] = (verts_list, faces_list, offset + 4)

    def _build(verts_list, faces_list, color):
        verts = np.concatenate(verts_list)
        faces = np.concatenate(faces_list)
        colors = np.tile(color, (len(verts), 1))
        return trimesh.Trimesh(
            vertices=verts, faces=faces, vertex_colors=colors, process=False
        )

    light = _build(groups[0][0], groups[0][1], GROUND_COLOR_A)
    dark = _build(groups[1][0], groups[1][1], GROUND_COLOR_B)
    return light, dark


def visualize_wis3d(
    output_dir: Path,
    object_mesh: trimesh.Trimesh,
    object_poses: np.ndarray,
    human_vertices: np.ndarray,
    human_faces: np.ndarray,
    cam_extrinsics: dict[str, np.ndarray] | None = None,
    ground_plane: np.ndarray | None = None,
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
        ground_plane: Optional (4,) plane coefficients [a, b, c, d].
    """
    n_frames = min(len(object_poses), len(human_vertices))
    human_colors = np.tile(HUMAN_MESH_COLOR, (human_vertices.shape[1], 1))

    ground_light = None
    ground_dark = None
    if ground_plane is not None:
        scene_center = human_vertices[:, :, :].mean(axis=(0, 1))
        ground_light, ground_dark = _make_ground_plane_meshes(
            ground_plane, center=scene_center
        )

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

        if ground_light is not None:
            scene.add_mesh("ground_light", ground_light)
            scene.add_mesh("ground_dark", ground_dark)

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

    ground_plane = None
    gp_path = cfg.get("ground_plane_path")
    if gp_path and Path(gp_path).exists():
        with open(gp_path) as f:
            ground_plane = np.array(json.load(f)["plane"])

    visualize_wis3d(
        output_dir=Path(cfg.output_dir),
        object_mesh=object_mesh,
        object_poses=object_poses,
        human_vertices=human_vertices,
        human_faces=human_faces,
        cam_extrinsics=cam_extrinsics,
        ground_plane=ground_plane,
    )


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Generate Wis3D visualization from config")
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--object_mesh_path", type=str, required=True)
    parser.add_argument("--object_pose_dir", type=str, required=True)
    parser.add_argument("--human_pose_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ground_plane_dir", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional override config (merged on top of defaults)")
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "mv_visualize_wis3d.yaml")
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))
    overrides = {k: v for k, v in vars(args).items() if k != "config_path" and v is not None}
    cfg = OmegaConf.merge(cfg, overrides)
    visualize_wis3d_from_config(cfg)
