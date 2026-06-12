from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from v2d.mv.rig import RigConfig

from v2d.mv.postprocess.lib.mv_eval_silhouette_mask import (
    mv_eval_silhouette_mask_rigid_object,
)


def mv_eval_silhouette_mask_object_from_config(cfg):
    """Load object mesh + poses and run 2D silhouette-vs-SAM2 evaluation."""
    rig = RigConfig(cfg.rig_config, camera_params_path=cfg.camera_params_path)

    canonical_mesh = trimesh.load(cfg.object_mesh_path, process=False, force="mesh")
    canonical_verts = np.asarray(canonical_mesh.vertices)
    faces = np.asarray(canonical_mesh.faces)
    poses = np.load(cfg.object_pose_path)

    cam_names: list[str] = []
    cam_intrinsics: list[np.ndarray] = []
    cam_extrinsics: list[np.ndarray] = []
    mask_dirs: list[Path] = []

    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)
        cam_names.append(cam.name)
        cam_intrinsics.append(cam.param.K)
        cam_extrinsics.append(cam.param.T)
        mask_dirs.append(Path(cfg.mask_path_template.format(cam_name=cam.name)))

    output_path = Path(cfg.output_path)
    eval_image_size = tuple(cfg.eval_image_size) if cfg.get("eval_image_size") else None
    debug = cfg.get("debug", 0)
    vis_dir = Path(cfg.vis_path) if cfg.get("vis_path") else None

    return mv_eval_silhouette_mask_rigid_object(
        cam_names=cam_names,
        cam_intrinsics=cam_intrinsics,
        cam_extrinsics=cam_extrinsics,
        mask_dirs=mask_dirs,
        canonical_verts=canonical_verts,
        faces=faces,
        poses=poses,
        output_path=output_path,
        eval_image_size=eval_image_size,
        erosion_kernel=int(cfg.get("erosion_kernel", 3)),
        erosion_iterations=int(cfg.get("erosion_iterations", 1)),
        min_mask_pixels=int(cfg.get("min_mask_pixels", 10)),
        debug=debug,
        vis_dir=vis_dir,
        tile_shape=tuple(cfg.get("tile_shape", [2, 2])),
        tile_image_size=tuple(cfg.get("tile_image_size", [768, 576])),
        camera_workers=int(cfg.get("camera_workers", 1)),
        profile=bool(cfg.get("profile", False)),
        progress_interval=float(cfg.get("progress_interval", 0.1)),
    )


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(
        description="2D silhouette-vs-SAM2 evaluation for object mesh"
    )
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--object_mesh_path", type=str, required=True)
    parser.add_argument("--object_pose_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(Path(__file__).parent / "mv_eval_silhouette_mask_object.yaml")
    if args.config_path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.config_path))
    overrides = {
        "camera_params_path": args.camera_params_path,
        "object_mesh_path": args.object_mesh_path,
        "object_pose_dir": args.object_pose_dir,
        "output_dir": args.output_dir,
        "mask_dir": args.mask_dir,
    }
    cfg = OmegaConf.merge(cfg, overrides)
    mv_eval_silhouette_mask_object_from_config(cfg)
