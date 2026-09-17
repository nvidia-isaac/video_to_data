from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import av
import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from lib_mhr import MHRLayer, MHR_PARAM_DIMS
from lib_mhr.human_texture import batched_vertex_normals
from prep.mhr_export_utils import camera_calibration, load_edex, read_rgb
from tools.pipeline_timing import PipelineTimer
from tools.mhr_mesh_renderer import NvdiffMeshRenderer


HUMAN_ALBEDO_HEX = "#C8D2D8"
HUMAN_ALBEDO = (200.0 / 255.0, 210.0 / 255.0, 216.0 / 255.0)
MESH_BACKGROUND_HEX = "#202428"
MESH_BACKGROUND = np.array([32, 36, 40], dtype=np.uint8)


def _params(block: Mapping[str, Any], indices: slice, device: torch.device) -> dict[str, torch.Tensor]:
    missing = [key for key in MHR_PARAM_DIMS if key not in block]
    if missing:
        raise KeyError(f"MHR inference bundle is missing parameters: {missing}")
    return {key: torch.as_tensor(np.asarray(block[key][indices]), device=device, dtype=torch.float) for key in MHR_PARAM_DIMS}


def _side_by_side_frame(rgb: np.ndarray, rendered: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    rendered = np.asarray(rendered, dtype=np.float32)
    foreground = np.asarray(foreground)
    if rendered.shape != rgb.shape or foreground.shape != rgb.shape[:2] or foreground.dtype != np.bool_:
        raise ValueError(f"RGB, render, and foreground must have [H,W,3], [H,W,3], and bool [H,W] shapes, got {rgb.shape}, {rendered.shape}, and {foreground.shape} {foreground.dtype}")
    mesh_panel = np.broadcast_to(MESH_BACKGROUND, rgb.shape).copy()
    mesh_panel[foreground] = np.rint(np.clip(rendered[foreground], 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.concatenate([rgb, mesh_panel], axis=1)


def _overlay_frame(rgb: np.ndarray, rendered: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    rendered = np.asarray(rendered, dtype=np.float32)
    foreground = np.asarray(foreground)
    if rendered.shape != rgb.shape or foreground.shape != rgb.shape[:2] or foreground.dtype != np.bool_:
        raise ValueError(f"RGB, render, and foreground must have [H,W,3], [H,W,3], and bool [H,W] shapes, got {rgb.shape}, {rendered.shape}, and {foreground.shape} {foreground.dtype}")
    rendered = np.rint(np.clip(rendered, 0.0, 1.0) * 255.0).astype(np.uint8)
    output = rgb.copy()
    output[foreground] = np.rint(rgb[foreground].astype(np.float32) * 0.28 + rendered[foreground].astype(np.float32) * 0.72).astype(np.uint8)
    return output


def _header(image: np.ndarray, text: str, height: int = 40) -> np.ndarray:
    bar = np.zeros((height, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate([bar, image], axis=0)


def _comparison_frame(rgb: np.ndarray, initial_render: np.ndarray, initial_mask: np.ndarray, predicted_render: np.ndarray, predicted_mask: np.ndarray, refined_render: np.ndarray, refined_mask: np.ndarray, frame_name: str, refined_title: str) -> np.ndarray:
    initial = _header(_overlay_frame(rgb, initial_render, initial_mask), "Initialization HOI")
    predicted = _header(_overlay_frame(rgb, predicted_render, predicted_mask), "CoCoNet prediction")
    refined = _header(_overlay_frame(rgb, refined_render, refined_mask), refined_title)
    return _header(np.concatenate([initial, predicted, refined], axis=1), f"frame ID {frame_name} | camera 0 | no ground truth used", height=44)


def _source_video_properties(video: Path) -> tuple[float, str, int, int, int]:
    container = av.open(str(video))
    stream = container.streams.video[0]
    frame_count = int(stream.frames or 0)
    if frame_count <= 0:
        frame_count = sum(1 for _ in container.decode(stream))
    rate = stream.average_rate
    fps = float(rate) if rate is not None else float("nan")
    rate_text = f"{rate.numerator}/{rate.denominator}" if rate is not None else ""
    width, height = int(stream.width), int(stream.height)
    container.close()
    if not np.isfinite(fps) or fps <= 0 or min(frame_count, width, height) <= 0:
        raise ValueError(f"Source video has invalid properties: fps={fps}, frames={frame_count}, size={width}x{height}")
    return fps, rate_text, frame_count, width, height


def render_mhr_wild_inference(export_seq: str | Path, source_video: str | Path, object_mesh: str | Path, before_bundle: str | Path, after_bundle: str | Path, output: str | Path, comparison_output: str | Path, *, device_name: str = "cuda", batch_size: int = 4, overwrite: bool = False, profiler: PipelineTimer | None = None) -> tuple[Path, Path]:
    setup_started = profiler.start() if profiler is not None else 0.0
    export_seq, source_video, object_mesh, before_bundle, after_bundle, output, comparison_output = map(lambda value: Path(value).resolve(), (export_seq, source_video, object_mesh, before_bundle, after_bundle, output, comparison_output))
    for path in (export_seq / "wild_export.json", source_video, object_mesh, before_bundle, after_bundle):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output == comparison_output:
        raise ValueError("Reconstruction and comparison outputs must use different paths")
    for path in (output, comparison_output):
        if path.exists() and not overwrite:
            raise FileExistsError(path)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not torch.cuda.is_available() and str(device_name).startswith("cuda"):
        raise RuntimeError("Wild MHR rendering requires CUDA")
    device = torch.device(device_name)
    before = torch.load(before_bundle, map_location="cpu", weights_only=False)
    after = torch.load(after_bundle, map_location="cpu", weights_only=False)
    names = [str(value) for value in before.get("frames", [])]
    if not names or [str(value) for value in after.get("frames", [])] != names:
        raise ValueError("CoCoNet/refined inference bundle timelines differ or are empty")
    if int(before.get("checkpoint", {}).get("step", -1)) != int(after.get("checkpoint", {}).get("step", -1)):
        raise ValueError("CoCoNet/refined inference bundles use different checkpoints")
    postopt_mode = str(after.get("postopt", {}).get("mode", ""))
    refined_title = "Contact-guided refinement" if postopt_mode == "smplh_parity" else "Refined result"
    K, world_to_camera = camera_calibration(load_edex(export_seq), 0)
    if not np.allclose(world_to_camera, np.eye(4), atol=1e-6):
        raise ValueError("Wild render expects camera space to equal export world space")
    first_rgb = read_rgb(export_seq, 0, names[0])
    height, width = first_rgb.shape[:2]
    source_fps, source_rate, source_frame_count, source_width, source_height = _source_video_properties(source_video)
    if len(names) != source_frame_count or (width, height) != (source_width, source_height):
        raise ValueError(f"Refined/export timeline differs from source video: bundle={len(names)} frames at {width}x{height}, source={source_frame_count} frames at {source_width}x{source_height}")
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=os.environ.get("MHR_ASSETS_ROOT"), device=str(device))
    faces_value = layer.mesh_faces(device=device)
    faces = faces_value.detach().cpu().numpy().astype(np.int32) if torch.is_tensor(faces_value) else np.asarray(faces_value, dtype=np.int32)
    renderer = NvdiffMeshRenderer(device=str(device))
    K_batch_template = np.asarray(K, dtype=np.float32)
    for path in (output, comparison_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.mp4")
    comparison_temporary = comparison_output.with_name(f".{comparison_output.name}.{os.getpid()}.tmp.mp4")
    writer = imageio.get_writer(str(temporary), format="FFMPEG", mode="I", fps=source_fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None, input_params=["-r", source_rate], ffmpeg_params=["-crf", "18", "-preset", "medium"])
    comparison_writer = imageio.get_writer(str(comparison_temporary), format="FFMPEG", mode="I", fps=source_fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None, input_params=["-r", source_rate], ffmpeg_params=["-crf", "18", "-preset", "medium"])
    if profiler is not None:
        profiler.record("setup", setup_started)
        profiler.update_metadata({"frames": len(names), "batch_size": int(batch_size), "height": height, "width": width, "layouts": ["source_rgb_and_refined_mesh", "initialization_coconet_refined_overlays"], "human_albedo": HUMAN_ALBEDO_HEX, "mesh_background": MESH_BACKGROUND_HEX})
    try:
        for start in tqdm(range(0, len(names), batch_size), desc="Render wild MHR inference"):
            stop = min(start + batch_size, len(names))
            indices = slice(start, stop)
            decode_started = profiler.start() if profiler is not None else 0.0
            with torch.inference_mode():
                initial_body = layer.mhr_forward(_params(before["in"], indices, device))
                predicted_body = layer.mhr_forward(_params(before["pr_initial"], indices, device))
                after_body = layer.mhr_forward(_params(after["pr"], indices, device))
            initial_vertices = initial_body.vertices.detach().cpu().numpy().astype(np.float32)
            predicted_vertices = predicted_body.vertices.detach().cpu().numpy().astype(np.float32)
            after_vertices = after_body.vertices.detach().cpu().numpy().astype(np.float32)
            initial_normals = batched_vertex_normals(initial_vertices, faces)
            predicted_normals = batched_vertex_normals(predicted_vertices, faces)
            after_normals = batched_vertex_normals(after_vertices, faces)
            if profiler is not None:
                profiler.record("body_decode_and_normals", decode_started)
            initial_poses = np.asarray(before["in"]["pose_abs"][indices], dtype=np.float32)
            predicted_poses = np.asarray(before["pr_initial"]["pose_abs"][indices], dtype=np.float32)
            after_poses = np.asarray(after["pr"]["pose_abs"][indices], dtype=np.float32)
            intrinsics = np.repeat(K_batch_template[None], stop - start, axis=0)
            render_started = profiler.start() if profiler is not None else 0.0
            initial_render, initial_mask = renderer.render_front_batch_constant_human_textured_object(initial_vertices, faces, initial_normals, str(object_mesh), initial_poses, intrinsics, (height, width), HUMAN_ALBEDO)
            predicted_render, predicted_mask = renderer.render_front_batch_constant_human_textured_object(predicted_vertices, faces, predicted_normals, str(object_mesh), predicted_poses, intrinsics, (height, width), HUMAN_ALBEDO)
            after_render, after_mask = renderer.render_front_batch_constant_human_textured_object(after_vertices, faces, after_normals, str(object_mesh), after_poses, intrinsics, (height, width), HUMAN_ALBEDO)
            if profiler is not None:
                profiler.record("mesh_render", render_started)
            compose_started = profiler.start() if profiler is not None else 0.0
            for local, frame_name in enumerate(names[start:stop]):
                rgb = read_rgb(export_seq, 0, frame_name)
                if rgb.shape != (height, width, 3):
                    raise ValueError(f"Wild RGB shape changed at frame {frame_name}: {rgb.shape}")
                writer.append_data(_side_by_side_frame(rgb, after_render[local], after_mask[local]))
                comparison_writer.append_data(_comparison_frame(rgb, initial_render[local], initial_mask[local], predicted_render[local], predicted_mask[local], after_render[local], after_mask[local], frame_name, refined_title))
            if profiler is not None:
                profiler.record("rgb_compose_and_encode", compose_started)
            print("MHR_WILD_RENDER_PROGRESS " + json.dumps({"current": stop, "total": len(names), "unit": "frames"}, sort_keys=True), flush=True)
    finally:
        finalize_started = profiler.start() if profiler is not None else 0.0
        writer.close()
        comparison_writer.close()
        if profiler is not None:
            profiler.record("video_writer_close", finalize_started)
    commit_started = profiler.start() if profiler is not None else 0.0
    os.replace(temporary, output)
    os.replace(comparison_temporary, comparison_output)
    if profiler is not None:
        profiler.record("output_commit", commit_started)
    return output, comparison_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render final reconstruction and initialization/CoCoNet/refinement comparison videos.")
    parser.add_argument("export_seq")
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--object-mesh", required=True)
    parser.add_argument("--before-bundle", required=True)
    parser.add_argument("--after-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--comparison-output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with PipelineTimer("wild_inference_render") as profiler:
        print(render_mhr_wild_inference(args.export_seq, args.source_video, args.object_mesh, args.before_bundle, args.after_bundle, args.output, args.comparison_output, device_name=args.device, batch_size=args.batch_size, overwrite=args.overwrite, profiler=profiler))


if __name__ == "__main__":
    main()
