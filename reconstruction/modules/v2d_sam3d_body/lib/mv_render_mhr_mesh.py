from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pyglet
pyglet.options['headless'] = True
import torch
from tqdm import tqdm

from v2d.mv.rig import RigConfig
from v2d.mv.io.video import FrameSource, get_video_writer
from .renderer import Renderer


def render_mhr_mesh(
    source: FrameSource,
    output_path: Path,
    pred_vertices: np.ndarray,
    pred_cam_t: np.ndarray,
    cam_intrinsics: np.ndarray,
    cam_extrinsics: np.ndarray,
    faces: np.ndarray,
):
    """Render MHR mesh overlay onto video frames.

    Args:
        source: Frame source (image dir or video).
        output_path: Output video path.
        pred_vertices: (N, V, 3) vertices in body frame (without pred_cam_t).
        pred_cam_t: (N, 3) world-frame translation.
        cam_intrinsics: (3, 3) camera intrinsic matrix.
        cam_extrinsics: (4, 4) T_world_from_camera matrix.
        faces: (F, 3) mesh face indices.
    """
    pred_vertices = pred_vertices + pred_cam_t[:, None, :]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = get_video_writer(output_path, fps=30, crf=23)
    with Renderer(image_size=source.image_size) as renderer:
        for i, image in enumerate(tqdm(source.iter_frames(), total=source.n_frames, desc="Rendering MHR mesh")):
            if i >= pred_vertices.shape[0]:
                break
            rendered_image = renderer.render_overlay(
                vertices=pred_vertices[i],
                faces=faces,
                K=cam_intrinsics,
                T=cam_extrinsics,
                image=image,
            ) * 255.0
            cv2.putText(rendered_image, f"Frame {i}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            writer.write_frame(rendered_image.astype(np.uint8))
    writer.close()
    print(f"Saved rendered video to {output_path}")


def render_from_config(cfg, rig: RigConfig):
    mhr_mesh = torch.load(cfg.mhr_mesh_mv_path)
    mhr_params = torch.load(cfg.mhr_params_mv_path)

    pred_vertices = mhr_mesh["pred_vertices"].cpu().numpy()
    faces = mhr_mesh["faces"].cpu().numpy()
    pred_cam_t = mhr_params["pred_cam_t"].cpu().numpy()

    for cam_id in cfg.cameras:
        cam = rig.get_camera(cam_id)
        image_dir = Path(cfg.image_path_template.format(cam_name=cam.name))

        cam_intrinsics = np.load(cfg.cam_intrinsics_path_template.format(cam_name=cam.name))
        cam_extrinsics = np.load(cfg.cam_extrinsics_path_template.format(cam_name=cam.name))

        source = FrameSource(image_dir=image_dir)
        output_path = Path(cfg.output_path) / f"mhr_mesh_mv_{cam.name}.mp4"

        render_mhr_mesh(
            source=source,
            output_path=output_path,
            pred_vertices=pred_vertices,
            pred_cam_t=pred_cam_t,
            cam_intrinsics=cam_intrinsics,
            cam_extrinsics=cam_extrinsics,
            faces=faces,
        )


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Render MHR mesh overlay for all cameras from config")
    parser.add_argument("--data_path", type=str, required=True, help="Root data directory")
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(Path(__file__).parent / "mv_optimize_mhr_params.yaml"),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)
    cfg = OmegaConf.merge(cfg, {"data_path": args.data_path})
    rig = RigConfig(cfg.rig_config)
    render_from_config(cfg, rig)
