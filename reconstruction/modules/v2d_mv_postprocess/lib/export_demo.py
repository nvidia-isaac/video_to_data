"""Demonstrate loading and visualizing the exported flat training data.

Loads calibration, meshes, poses, params, depth, masks from the flat layout
produced by export_sequence.py and renders HOI overlays as a sanity check.

Usage:
    python -m v2d.mv.postprocess.lib.export_demo \
        --seq_dir /local/path/to/sequence \
        --output_dir /local/path/to/output
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import pyglet
pyglet.options["headless"] = True
import torch
import trimesh
import yaml
from tqdm import tqdm

from v2d.common.datatypes import DepthImage
from v2d.mv.rig import RigConfig
from v2d.mv.io.video import FrameSource, get_video_writer, tile_videos
from v2d.mv.vis.renderer import Renderer

LEFT_CAMERAS = [
    "front_stereo_camera_left",
    "back_stereo_camera_left",
    "left_stereo_camera_left",
    "right_stereo_camera_left",
]

HUMAN_MESH_COLOR = np.array([102, 230, 179], dtype=np.uint8)


def export_demo(seq_dir: Path, output_dir: Path) -> None:
    """Load exported data and render HOI overlay as a sanity check."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Export Demo: loading data from flat layout")
    print(f"  seq_dir:    {seq_dir}")
    print(f"  output_dir: {output_dir}")
    print("=" * 60)

    # --- Load calibration ---
    edex_path = seq_dir / "edex"
    rig = RigConfig("stereo-4", camera_params_path=edex_path)
    print(f"\nCalibration loaded from {edex_path}")

    # --- Load object mesh ---
    mesh_files = glob.glob(str(seq_dir / "object_template" / "*"))
    mesh_files = [f for f in mesh_files if not f.endswith(".npy")]
    if not mesh_files:
        raise FileNotFoundError(f"No mesh files found in {seq_dir / 'object_template'}")
    object_mesh_path = mesh_files[0]
    object_mesh = trimesh.load(object_mesh_path, process=False, force="mesh")
    print(f"Object mesh: {object_mesh_path} ({len(object_mesh.vertices)} verts, {len(object_mesh.faces)} faces)")

    # --- Check for symmetry transforms ---
    sym_files = glob.glob(str(seq_dir / "object_template" / "symmetry_tfs*"))
    if sym_files:
        print(f"Symmetry transforms: {sym_files[0]}")
    else:
        print("Symmetry transforms: not yet available")

    # --- Load object poses ---
    poses_path = seq_dir / "poses.npy"
    object_poses = np.load(str(poses_path))
    print(f"Object poses: {poses_path} shape={object_poses.shape}")

    # --- Load human mesh ---
    mhr_mesh_path = seq_dir / "mhr_mesh_mv.pt"
    mhr_mesh = torch.load(str(mhr_mesh_path), weights_only=False, map_location="cpu")
    human_vertices = mhr_mesh["pred_vertices"].cpu().numpy()
    human_faces = mhr_mesh["faces"].cpu().numpy()
    print(f"Human mesh: vertices={human_vertices.shape} faces={human_faces.shape}")

    # --- Load human params ---
    mhr_params_path = seq_dir / "mhr_params_mv.pt"
    mhr_params = torch.load(str(mhr_params_path), weights_only=False, map_location="cpu")
    print(f"Human params keys: {list(mhr_params.keys())}")
    for k, v in mhr_params.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")

    # --- Load SOMA params ---
    soma_path = seq_dir / "soma_params.npz"
    if soma_path.exists():
        soma = np.load(str(soma_path))
        print(f"SOMA params keys: {list(soma.keys())}")
        for k in soma.keys():
            print(f"  {k}: shape={soma[k].shape} dtype={soma[k].dtype}")
    else:
        print("SOMA params: not found (skipping)")

    # --- Load ground plane ---
    gp_path = seq_dir / "ground_plane.json"
    if gp_path.exists():
        with open(gp_path) as f:
            ground_plane = json.load(f)
        print(f"Ground plane: normal={ground_plane.get('normal')}, offset={ground_plane.get('offset')}")
    else:
        print("Ground plane: not found (skipping)")

    # --- Load HOI metadata ---
    meta_path = seq_dir / "hoi_metadata.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            hoi_meta = yaml.safe_load(f)
        obj_id = hoi_meta.get("object", {}).get("id", "unknown")
        prompt = hoi_meta.get("object", {}).get("prompt", "")
        print(f"HOI metadata: object_id={obj_id}, prompt='{prompt}'")
    else:
        print("HOI metadata: not found")

    # --- Per-camera data check and rendering ---
    print("\n--- Per-camera data ---")
    overlay_paths: list[Path] = []
    cam_names: list[str] = []
    human_colors = np.tile(HUMAN_MESH_COLOR, (human_vertices.shape[1], 1))

    for cam_name in LEFT_CAMERAS:
        cam = rig.get_camera_by_name(cam_name)
        if cam is None:
            print(f"  {cam_name}: not found in rig, skipping")
            continue

        image_dir = seq_dir / "images" / cam_name
        if not image_dir.exists():
            print(f"  {cam_name}: no images dir, skipping")
            continue

        source = FrameSource(image_dir=image_dir)
        n_rgb = source.n_frames
        print(f"\n  {cam_name}:")
        print(f"    RGB frames: {n_rgb}  resolution: {source.image_size}")

        # Depth sample
        depth_dir = seq_dir / "depth" / cam_name
        if depth_dir.exists():
            depth_files = sorted(depth_dir.glob("*.png"))
            n_depth = len(depth_files)
            if depth_files:
                sample_depth = DepthImage.load(str(depth_files[0]))
                print(f"    Depth frames: {n_depth}  sample shape={sample_depth.depth.shape} "
                      f"min={sample_depth.depth.min():.3f} max={sample_depth.depth.max():.3f}")
            else:
                print(f"    Depth frames: 0")
        else:
            print(f"    Depth: directory not found")

        # Masks
        obj_mask_dir = seq_dir / "object_masks" / cam_name
        human_mask_dir = seq_dir / "human_masks" / cam_name
        n_obj_masks = len(list(obj_mask_dir.glob("*.png"))) if obj_mask_dir.exists() else 0
        n_human_masks = len(list(human_mask_dir.glob("*.png"))) if human_mask_dir.exists() else 0
        print(f"    Object masks: {n_obj_masks}")
        print(f"    Human masks:  {n_human_masks}")

        if n_rgb != n_obj_masks:
            print(f"    WARNING: RGB/object mask count mismatch ({n_rgb} vs {n_obj_masks})")
        if n_rgb != n_human_masks:
            print(f"    WARNING: RGB/human mask count mismatch ({n_rgb} vs {n_human_masks})")

        # Render HOI overlay
        n_frames = min(n_rgb, len(object_poses), len(human_vertices))
        overlay_path = output_dir / f"{cam_name}_hoi_overlay.mp4"
        print(f"    Rendering overlay ({n_frames} frames) -> {overlay_path}")

        writer = get_video_writer(overlay_path, fps=30, crf=23)
        with Renderer(image_size=source.image_size) as renderer:
            for i, image in enumerate(tqdm(
                source.iter_frames(), total=source.n_frames,
                desc=f"    {cam_name}", leave=False,
            )):
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

                rendered = renderer.render_overlay(
                    meshes=[obj_mesh_i, human_mesh_i],
                    K=cam.param.K,
                    T=cam.param.T,
                    image=image,
                ) * 255.0

                frame_text = f"Frame {i}"
                (tw, th), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.putText(
                    rendered, frame_text,
                    (rendered.shape[1] - tw - 10, th + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )
                writer.write_frame(rendered.astype(np.uint8))
        writer.close()
        overlay_paths.append(overlay_path)
        cam_names.append(cam_name)

    # Tile overlays
    if len(overlay_paths) >= 2:
        tiled_path = output_dir / "tiled_demo.mp4"
        print(f"\nTiling {len(overlay_paths)} overlays -> {tiled_path}")
        tile_videos(
            sources=[FrameSource(video_path=p) for p in overlay_paths],
            output_path=tiled_path,
            tile_shape=(2, 2),
            video_names=cam_names,
        )
        print(f"Saved tiled demo to {tiled_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Data Summary:")
    print(f"  Cameras:          {len(cam_names)} ({', '.join(cam_names)})")
    print(f"  Object mesh:      {object_mesh_path}")
    print(f"  Object poses:     {object_poses.shape}")
    print(f"  Human vertices:   {human_vertices.shape}")
    print(f"  Human faces:      {human_faces.shape}")
    print(f"  Human param keys: {list(mhr_params.keys())}")
    if soma_path.exists():
        print(f"  SOMA param keys:  {list(np.load(str(soma_path)).keys())}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo: load and visualize exported training data")
    parser.add_argument("--seq_dir", type=str, required=True,
                        help="Path to the exported sequence directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to write demo output (overlay videos)")
    args = parser.parse_args()

    export_demo(seq_dir=Path(args.seq_dir), output_dir=Path(args.output_dir))
