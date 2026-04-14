from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pyglet
pyglet.options['headless'] = True
import torch
import trimesh
from tqdm import tqdm

from v2d.mv.rig import RigConfig
from v2d.mv.io.video import FrameSource, get_video_writer, tile_videos
from v2d.mv.vis.renderer import Renderer

HUMAN_MESH_COLOR = np.array([102, 230, 179], dtype=np.uint8)  # light green


def render_hoi_overlay(
    source: FrameSource,
    output_path: Path,
    object_mesh: trimesh.Trimesh,
    object_poses: np.ndarray,
    human_vertices: np.ndarray,
    human_faces: np.ndarray,
    cam_intrinsics: np.ndarray,
    cam_extrinsics: np.ndarray,
):
    """Render object + human mesh overlay onto video frames.

    Args:
        source: Frame source (image dir or video).
        output_path: Output video path.
        object_mesh: Object mesh in its canonical frame.
        object_poses: (N, 4, 4) per-frame object-to-world poses.
        human_vertices: (N, V, 3) human vertices in world frame.
        human_faces: (F, 3) human mesh face indices.
        cam_intrinsics: (3, 3) camera intrinsic matrix.
        cam_extrinsics: (4, 4) T_world_from_camera matrix.
    """
    n_frames = min(source.n_frames, len(object_poses), len(human_vertices))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = get_video_writer(output_path, fps=30, crf=23)
    human_colors = np.tile(HUMAN_MESH_COLOR, (human_vertices.shape[1], 1))

    with Renderer(image_size=source.image_size) as renderer:
        for i, image in enumerate(tqdm(source.iter_frames(), total=source.n_frames, desc="Rendering HOI overlay")):
            if i >= n_frames:
                break

            obj_mesh_i = object_mesh.copy()
            obj_mesh_i.apply_transform(object_poses[i])

            human_mesh_i = trimesh.Trimesh(
                vertices=human_vertices[i],
                faces=human_faces,
                vertex_colors=human_colors,
                process=False,
            )

            rendered_image = renderer.render_overlay(
                meshes=[obj_mesh_i, human_mesh_i],
                K=cam_intrinsics,
                T=cam_extrinsics,
                image=image,
            ) * 255.0
            frame_text = f"Frame {i}"
            (tw, th), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(rendered_image, frame_text, (rendered_image.shape[1] - tw - 10, th + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            writer.write_frame(rendered_image.astype(np.uint8))
    writer.close()
    print(f"Saved HOI overlay video to {output_path}")


def render_hoi_overlay_from_config(cfg):
    rig = RigConfig(cfg.rig_config, camera_params_path=cfg.camera_params_path)

    object_mesh = trimesh.load(cfg.object_mesh_path, process=False, force='mesh')

    mhr_mesh = torch.load(cfg.mhr_mesh_mv_path, weights_only=False, map_location="cpu")

    pred_vertices = mhr_mesh["pred_vertices"].cpu().numpy()
    human_faces = mhr_mesh["faces"].cpu().numpy()
    pred_cam_t = mhr_mesh["pred_cam_t"].cpu().numpy()
    human_vertices = pred_vertices + pred_cam_t[:, None, :]

    object_poses = np.load(cfg.object_pose_path)

    overlay_paths: list[Path] = []
    cam_names: list[str] = []

    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)

        if cfg.get("image_dir"):
            source = FrameSource(image_dir=Path(cfg.image_path_template.format(cam_name=cam.name)))
        else:
            source = FrameSource(video_path=Path(cfg.video_path_template.format(cam_name=cam.name)))

        output_path = Path(cfg.output_dir) / f"{cam.name}_hoi_overlay.mp4"

        print(f"Rendering HOI overlay for camera {cam.name}...")
        render_hoi_overlay(
            source=source,
            output_path=output_path,
            object_mesh=object_mesh,
            object_poses=object_poses,
            human_vertices=human_vertices,
            human_faces=human_faces,
            cam_intrinsics=cam.param.K,
            cam_extrinsics=cam.param.T,
        )
        overlay_paths.append(output_path)
        cam_names.append(cam.name)

    tile_shape = tuple(cfg.get("tile_shape", [2, 2]))
    tile_image_size = tuple(cfg.get("tile_image_size", None))
    tiled_path = Path(cfg.output_dir) / "tiled_hoi_overlay.mp4"
    print(f"Tiling {len(overlay_paths)} overlays into {tiled_path}...")
    tile_videos(
        sources=[FrameSource(video_path=p) for p in overlay_paths],
        output_path=tiled_path,
        tile_shape=tile_shape,
        output_image_size=tile_image_size,
        video_names=cam_names,
    )
    print(f"Saved tiled HOI overlay to {tiled_path}")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Render HOI overlay for all cameras from config")
    parser.add_argument("--camera_params_path", type=str, required=True)
    parser.add_argument("--object_mesh_path", type=str, required=True)
    parser.add_argument("--object_pose_dir", type=str, required=True)
    parser.add_argument("--human_pose_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(Path(__file__).parent / "mv_render_hoi_overlay.yaml"),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    overrides = {k: v for k, v in vars(args).items() if k != "config_path" and v is not None}
    cfg = OmegaConf.merge(cfg, overrides)
    render_hoi_overlay_from_config(cfg)
